//! The DeepSeek-V4.1-Flash engine thread: continuous batching over the engine's sequence slots. Every step runs
//! one window: a token for each decoding request, then prompt chunks for the requests still reading theirs; the
//! samples are drawn on the host (structured outputs mask the logits first) and streamed back as text deltas.
use crate::worker::{Detok, Event, Job};
use anyhow::{anyhow, Result};
use qx_engine::dsv41::{DsEngine, DS_TMAX};
use qx_engine::engine::Sampling;
use qx_pack::gguf::Gguf;
use qx_tok::Tok;
use std::sync::mpsc;

/// Sets the logits of the tokens a structured-output mask disallows (bit i of the words clear) to -inf.
fn apply_mask(logits: &mut [f32], mask: &[u32]) {
    for (i, v) in logits.iter_mut().enumerate() {
        let allowed = mask.get(i / 32).is_some_and(|w| w >> (i % 32) & 1 == 1);
        if !allowed {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Temperature, top-k, top-p, min-p over one token's logits, then a draw keyed by (seed, position); temperature 0
/// (or top-k 1) is argmax.
pub fn sample(logits: &[f32], sp: &Sampling, pos: usize) -> u32 {
    let argmax = |x: &[f32]| x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    if sp.temperature <= 0.0 || sp.top_k == 1 {
        return argmax(logits);
    }
    let k = (sp.top_k.max(1) as usize).min(logits.len());
    let mut cand: Vec<(f32, u32)> = logits.iter().enumerate().map(|(i, &v)| (v / sp.temperature, i as u32)).collect();
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(k);
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let m = cand[0].0;
    let mut probs: Vec<f32> = cand.iter().map(|c| (c.0 - m).exp()).collect();
    let z: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= z);
    // top-p: the smallest prefix whose mass reaches top_p; min-p: nothing under min_p times the best
    let mut keep = probs.len();
    if sp.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= sp.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    if sp.min_p > 0.0 {
        let floor = sp.min_p * probs[0];
        keep = keep.min(probs.iter().take_while(|&&p| p >= floor).count().max(1));
    }
    probs.truncate(keep);
    let z: f32 = probs.iter().sum();
    // splitmix64 of (seed, position)
    let mut x = sp.seed ^ (pos as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    let u = (x >> 11) as f64 / (1u64 << 53) as f64;
    let mut acc = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += (p / z) as f64;
        if u < acc {
            return cand[i].1;
        }
    }
    cand[keep - 1].1
}

/// Window widths a step may use: capturing a width's graph costs about a second the first time, so the steps
/// draw from this set (the decoders, then prompt tokens up to the largest width the prompts can fill) and the
/// warm-up captures the common ones.
const WIDTHS: [usize; 14] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];

pub struct Options {
    pub max_ctx: usize,
    /// Tokens per prompt chunk (at most DS_TMAX).
    pub window: usize,
    /// Sequence slots (requests served at once).
    pub slots: usize,
    pub pack: std::path::PathBuf,
    pub vram_gib: f64,
    pub ram_gib: f64,
    pub profile: Option<std::path::PathBuf>,
    /// Where the live ranking is written after requests (at most once a minute): the profile to load next time.
    pub profile_out: Option<std::path::PathBuf>,
}

/// A request holding a slot.
struct Req<'a> {
    job: Job,
    slot: usize,
    detok: Detok<'a>,
    text: String,
    /// Tokens passed on.
    n: usize,
    t0: std::time::Instant,
    t_prompt: Option<f64>,
    finish: Option<&'static str>,
    matcher: Option<llguidance::Matcher>,
    /// Prompt tokens fed so far (the reused prefix counts).
    read: usize,
    reused: usize,
    /// The token the next step feeds (the last sample) once the prompt is read.
    next: Option<u32>,
    /// The distribution to sample from (after the step that ran the prompt's last token or a sample).
    logits: Option<Vec<f32>>,
}

impl Req<'_> {
    /// Passes a sampled token on; false once the request is finished.
    fn emit(&mut self, id: u32, stop_ids: &[u32], max_ctx: usize, pos: usize) -> bool {
        if stop_ids.contains(&id) {
            self.finish = Some("stop");
            return false;
        }
        if let Some(delta) = self.detok.push(id) {
            self.text.push_str(&delta);
            if let Some(st) = self.job.stop.iter().find(|st| !st.is_empty() && self.text.contains(st.as_str())) {
                let cut = self.text.find(st.as_str()).unwrap();
                let before = self.text.len() - delta.len();
                if cut > before {
                    let _ = self.job.events.send(Event::Delta(self.text[before..cut].to_string()));
                }
                self.finish = Some("stop");
                return false;
            }
            if self.job.events.send(Event::Delta(delta)).is_err() {
                self.finish = Some("cancelled");
                return false;
            }
        }
        self.n += 1;
        if self.n >= self.job.max_tokens || pos + 1 >= max_ctx {
            self.finish = Some("length");
            return false;
        }
        true
    }

    fn done(&self, stats: bool, tiers: &str) {
        let all = self.t0.elapsed().as_secs_f64();
        let tp = self.t_prompt.unwrap_or(all);
        let finish = self.finish.unwrap_or("length");
        if stats {
            eprintln!("req: slot {} {} prompt tokens ({} reused) in {:.2}s, {} generated in {:.2}s, {finish}{tiers}", self.slot, self.job.prompt.len(), self.reused, tp, self.n, all - tp);
        }
        let _ = self.job.events.send(Event::Done {
            prompt_tokens: self.job.prompt.len(),
            completion_tokens: self.n,
            finish,
            prompt_tps: self.job.prompt.len() as f64 / tp.max(1e-9),
            decode_tps: self.n as f64 / (all - tp).max(1e-9),
        });
    }
}

pub fn start(_g: &'static Gguf, tok: &'static Tok, opt: Options) -> Result<mpsc::Sender<Job>> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let pack: &'static Gguf = Box::leak(Box::new(Gguf::open(&opt.pack).map_err(|e| anyhow!("pack {}: {e}", opt.pack.display()))?));
    std::thread::spawn(move || {
        let mut eng = match DsEngine::load_tiered_slots(pack, opt.max_ctx, opt.vram_gib, opt.ram_gib, opt.profile.as_deref(), opt.slots) {
            Ok(e) => {
                let _ = ready_tx.send(Ok(()));
                e
            }
            Err(e) => {
                let _ = ready_tx.send(Err(format!("{e:#}")));
                return;
            }
        };
        let stop_ids: Vec<u32> = ["<｜end▁of▁sentence｜>"].iter().filter_map(|s| tok.specials.get(*s).copied()).chain(std::iter::once(tok.eos)).collect();
        let window = opt.window.clamp(1, DS_TMAX);
        let n_vocab = eng.cfg.n_vocab;
        let grammars = match crate::grammar::Grammars::new(tok, n_vocab) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("structured outputs unavailable: {e:#}");
                None
            }
        };
        // warm-up: the kernels' first compilation and the graph captures of the common widths (one token per
        // decoding slot, and the prompt chunk); the other widths of WIDTHS are captured on first use
        {
            let t0 = std::time::Instant::now();
            // one token per decoding slot, and the prompt chunk widths up to `window`
            let mut widths: Vec<usize> = (1..=eng.slots.min(4)).collect();
            let chunks: Vec<usize> = WIDTHS.iter().copied().filter(|&w| w >= 8 && w <= window && !widths.contains(&w)).collect();
            widths.extend(chunks);
            if !widths.contains(&window) {
                widths.push(window);
            }
            let toks = vec![0u32; 2 * DS_TMAX];
            eng.reset(0);
            let _ = eng.forward_window(&toks[..2], None); // positions 0-1: the stores' first rows, before any capture
            for &w in &widths {
                let _ = eng.forward_window(&toks[..w], None);
            }
            eng.reset(0);
            eprintln!("quixiinfer: warm-up {:.1}s (graphs for windows of {widths:?})", t0.elapsed().as_secs_f64());
        }
        let mut width_hist: std::collections::BTreeMap<usize, usize> = Default::default();
        let prefix_on = !std::env::var("QX_DS_PREFIX").is_ok_and(|v| v == "0");
        let stats = std::env::var("QX_SCHED_STATS").is_ok();
        let group = eng.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
        let mut last_save: Option<std::time::Instant> = None;
        let mut free: Vec<usize> = (0..eng.slots).rev().collect();
        let mut waiting: std::collections::VecDeque<Job> = Default::default();
        let mut running: Vec<Req> = Vec::new();
        let mut steps = 0usize;
        loop {
            if running.is_empty() && waiting.is_empty() {
                match rx.recv() {
                    Ok(j) => waiting.push_back(j),
                    Err(_) => return,
                }
            }
            while let Ok(j) = rx.try_recv() {
                waiting.push_back(j);
            }
            // admit: a free slot each, the one sharing the longest prefix with the prompt when there is one
            while !free.is_empty() {
                let Some(job) = waiting.pop_front() else { break };
                if job.prompt.is_empty() || job.prompt.len() + 1 >= opt.max_ctx {
                    let _ = job.events.send(Event::Error(format!("prompt of {} tokens does not fit the {}-token context", job.prompt.len(), opt.max_ctx)));
                    continue;
                }
                let matcher = match (&job.grammar, &grammars) {
                    (Some(spec), Some(gr)) => match gr.matcher(spec) {
                        Ok(m) => Some(m),
                        Err(e) => {
                            let _ = job.events.send(Event::Error(format!("{e:#}")));
                            continue;
                        }
                    },
                    (Some(_), None) => {
                        let _ = job.events.send(Event::Error("structured outputs unavailable (tokenizer trie)".into()));
                        continue;
                    }
                    (None, _) => None,
                };
                let shared = |slot: usize| -> usize {
                    let mut l = eng.tokens(slot).iter().zip(&job.prompt).take_while(|(a, b)| a == b).count().min(job.prompt.len() - 1);
                    l -= l % group;
                    if l >= 64 { l } else { 0 }
                };
                let (i, reuse) = if prefix_on { free.iter().enumerate().map(|(i, &sl)| (i, shared(sl))).max_by_key(|&(_, l)| l).unwrap() } else { (free.len() - 1, 0) };
                let slot = free.remove(i);
                if reuse > 0 {
                    eng.rewind(slot, reuse);
                } else {
                    eng.reset(slot);
                }
                running.push(Req { job, slot, detok: Detok { tok, ids: Vec::new(), sent: 0 }, text: String::new(), n: 0, t0: std::time::Instant::now(), t_prompt: None, finish: None, matcher, read: reuse, reused: reuse, next: None, logits: None });
            }
            if running.is_empty() {
                continue;
            }
            // the window: decoders first (one token each), then prompt tokens up to the largest width of WIDTHS
            // the prompts can fill (at most `window` per prompt)
            let mut segs: Vec<(usize, u32)> = Vec::new();
            let mut plan: Vec<(usize, usize, bool)> = Vec::new(); // (request index, tokens, completes the prompt or decodes)
            for (ri, r) in running.iter().enumerate() {
                if let Some(id) = r.next {
                    segs.push((r.slot, id));
                    plan.push((ri, 1, true));
                }
            }
            let d = segs.len();
            let avail: usize = running.iter().filter(|r| r.next.is_none() && r.read < r.job.prompt.len()).map(|r| (r.job.prompt.len() - r.read).min(window)).sum();
            let need = (d + avail).min(DS_TMAX);
            let width = WIDTHS.iter().copied().filter(|&w| w <= need && w <= DS_TMAX).max().unwrap_or(d).max(d);
            let mut room = width - d;
            for (ri, r) in running.iter().enumerate() {
                if room == 0 {
                    break;
                }
                if r.next.is_none() && r.read < r.job.prompt.len() {
                    let n = (r.job.prompt.len() - r.read).min(window).min(room);
                    segs.extend(r.job.prompt[r.read..r.read + n].iter().map(|&t| (r.slot, t)));
                    plan.push((ri, n, r.read + n == r.job.prompt.len()));
                    room -= n;
                }
            }
            if segs.is_empty() {
                continue;
            }
            *width_hist.entry(segs.len()).or_default() += 1;
            let out = eng.forward_segments(&segs, None);
            steps += 1;
            let mut row = 0;
            for &(ri, n, samples) in &plan {
                let r = &mut running[ri];
                if r.next.is_none() {
                    r.read += n;
                    if r.read == r.job.prompt.len() {
                        r.t_prompt = Some(r.t0.elapsed().as_secs_f64());
                    }
                }
                if samples {
                    r.logits = Some(out[(row + n - 1) * n_vocab..(row + n) * n_vocab].to_vec());
                }
                row += n;
            }
            // sample for every request with a distribution
            for r in running.iter_mut() {
                let Some(mut logits) = r.logits.take() else { continue };
                let pos = eng.pos(r.slot);
                if let Some(m) = r.matcher.as_mut() {
                    match m.compute_mask() {
                        Ok(mask) => apply_mask(&mut logits, mask.as_slice()),
                        Err(e) => {
                            let _ = r.job.events.send(Event::Error(format!("grammar: {e}")));
                            r.finish = Some("error");
                            continue;
                        }
                    }
                }
                let id = sample(&logits, &r.job.sampling, pos);
                if let Some(m) = r.matcher.as_mut() {
                    if !stop_ids.contains(&id) {
                        if let Err(e) = m.consume_token(id) {
                            let _ = r.job.events.send(Event::Error(format!("grammar: {e}")));
                            r.finish = Some("error");
                            continue;
                        }
                    }
                }
                if r.emit(id, &stop_ids, opt.max_ctx, pos) {
                    if r.matcher.as_ref().is_some_and(|m| m.is_stopped()) {
                        r.finish = Some("stop");
                    } else {
                        r.next = Some(id);
                    }
                }
            }
            // finished requests give their slots back (their tokens stay for a prefix match)
            let mut i = 0;
            while i < running.len() {
                if running[i].finish.is_some() {
                    let r = running.remove(i);
                    r.done(stats, &eng.tier_stats().unwrap_or_default());
                    free.push(r.slot);
                } else {
                    i += 1;
                }
            }
            if stats && (steps % 200 == 0 || running.is_empty()) {
                eprintln!("sched: {steps} steps, {} running, {} waiting, window widths {:?}", running.len(), waiting.len(), width_hist);
            }
            if let Some(path) = &opt.profile_out {
                if running.is_empty() && last_save.is_none_or(|t| t.elapsed().as_secs() >= 60) {
                    match eng.save_profile(path) {
                        Ok(()) => last_save = Some(std::time::Instant::now()),
                        Err(e) => eprintln!("profile not saved: {e:#}"),
                    }
                }
            }
        }
    });
    ready_rx.recv().map_err(|_| anyhow!("engine thread ended"))?.map_err(|e| anyhow!(e))?;
    Ok(tx)
}
