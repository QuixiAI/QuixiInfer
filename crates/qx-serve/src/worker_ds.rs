//! The DeepSeek-V4.1-Flash engine thread: continuous batching over the engine's sequence slots. A step is one
//! window (a token per decoding request, then prompt tokens) or one chunk of a long prompt stretch; the
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
    /// The constraint waits for the reasoning block to close (`Job::grammar_after_think`).
    in_think: bool,
    /// Prompt tokens fed so far (the reused prefix counts).
    read: usize,
    reused: usize,
    /// The token the next step feeds (the last sample) once the prompt is read.
    next: Option<u32>,
    /// The distribution to sample from (after the step that ran the prompt's last token or a sample).
    logits: Option<Vec<f32>>,
    /// Prompt-lookup speculation (QX_DS_SPEC): tokens already passed on that the next step feeds again before
    /// `next` (after a rejected draft the slot is rewound to a group boundary), the drafts the last step fed after
    /// `next`, the row of `next` in the step's output, the draft length to try, and steps to sit out.
    refeed: Vec<u32>,
    drafts: Vec<u32>,
    srow: Option<usize>,
    /// Per kind of draft (the slot's own history or the corpus of other replies, and an agreement under 16 tokens
    /// or of 16 and more: 0 own short, 1 own long, 2 corpus short, 3 corpus long): the draft length to try, the
    /// steps to sit out, what the kind's drafts have been worth lately and how often in a row that fell under
    /// zero. Structure (a list's keys) agrees shortly and fails at the values; a list with other keys agrees long
    /// and fails at each key; a copy agrees long and holds: each kind stands on its own record. `spec_src`: the
    /// kind of the last step's drafts.
    spec: [SpecSrc; 4],
    spec_src: usize,
}

/// One source's standing with a request. `gain` is in draft rows: a step's accepted tokens count 2 each (a token
/// costs a step, a draft row about half of one; beside other decoders 1.33), its draft rows and the token fed
/// again after a rejection 1 each; 0.7 of the old value and 0.3 of the last step's.
#[derive(Clone, Copy)]
struct SpecSrc {
    k: usize,
    cool: usize,
    gain: f32,
    trips: u32,
}
const SPEC_NEW: SpecSrc = SpecSrc { k: 2, cool: 0, gain: 1.0, trips: 0 };

/// Drafts by suffix: the place in `seq` whose preceding tokens agree with the end of `suf` over the most tokens
/// (at least `n`, at most 24 counted; the latest such place on a tie), and the up to `k` tokens that follow it
/// there. `SEP` (between the replies of the corpus) never matches and ends a continuation. Returns the agreed
/// length and the drafts; (0, empty) when there is none. The search stops with the best place so far after
/// 200,000 token comparisons (a history that is one token repeated agrees almost everywhere).
const SEP: u32 = u32::MAX;
fn suffix_draft(seq: &[u32], suf: &[u32], n: usize, k: usize) -> (usize, Vec<u32>) {
    let Some(&last) = suf.last() else { return (0, Vec::new()) };
    let n = n.min(24);
    if k == 0 || suf.len() < n || seq.len() < n + 1 {
        return (0, Vec::new());
    }
    let (mut best, mut at) = (0usize, 0usize);
    let mut work = 0usize;
    for e in (n - 1..seq.len() - 1).rev() {
        if seq[e] != last || seq[e + 1] == SEP {
            continue;
        }
        let m = (0..suf.len().min(24).min(e + 1)).take_while(|&j| seq[e - j] == suf[suf.len() - 1 - j]).count();
        work += m + 1;
        if work > 200_000 && m <= best {
            break;
        }
        if m > best {
            best = m;
            at = e;
            if m == suf.len().min(24) {
                break;
            }
        }
    }
    if best < n {
        return (0, Vec::new());
    }
    (best, seq[at + 1..seq.len().min(at + 1 + k)].iter().copied().take_while(|&t| t != SEP).collect())
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

const ENGINE_THREAD: &str = "qx-ds-engine";

/// MemAvailable of /proc/meminfo in bytes.
fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    Some(l.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
}

/// Gives the RAM tier's elastic part (QX_DS_RAM_YIELD GiB, `DsEngine::ram_yield`) back to the host while something
/// else needs the memory, and takes it again afterwards. Out when the path QX_DS_YIELD_PATH exists (a virtual
/// machine's pid file: known before the guest touches its memory) or the host's available memory is under
/// QX_DS_YIELD_LOW GiB (default 4); back when the path is gone and the available memory has stayed above the
/// part's size plus QX_DS_YIELD_HIGH GiB (default 8) for QX_DS_YIELD_HOLD seconds (default 20). Replies are the
/// same either way: the records that fall out are read from NVMe meanwhile.
struct Yielder {
    bytes: u64,
    path: Option<std::path::PathBuf>,
    low: u64,
    high: u64,
    hold: std::time::Duration,
    out: bool,
    checked: std::time::Instant,
    roomy_since: Option<std::time::Instant>,
    /// Records the last restore has yet to read back (reported once it reaches 0).
    refilling: bool,
    /// After a release: when it happened and what was available before, for the line that says what the host got.
    released: Option<(std::time::Instant, u64)>,
}

impl Yielder {
    fn new(eng: &DsEngine) -> Option<Self> {
        let bytes = eng.ram_elastic() as u64;
        if bytes == 0 {
            return None;
        }
        let gib = |k: &str, d: f64| (std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d) * 1073741824.0) as u64;
        let path = std::env::var("QX_DS_YIELD_PATH").ok().filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        let hold = std::time::Duration::from_secs_f64(std::env::var("QX_DS_YIELD_HOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0));
        let low = gib("QX_DS_YIELD_LOW", 4.0);
        // the part returns only into room that leaves the host well above the low mark (or it would go out again)
        let high = gib("QX_DS_YIELD_HIGH", 8.0).max(low + (2 << 30));
        // (a server started beside the path has the part out already)
        let out = eng.ram_yield_out();
        let me = Self { bytes, path, low, high, hold, out, checked: std::time::Instant::now(), roomy_since: None, refilling: false, released: None };
        eprintln!(
            "ram tier: {:.1} GiB go back to the host while {} or under {:.1} GiB are available, and return {:.0} s after",
            bytes as f64 / 1073741824.0,
            me.path.as_ref().map(|p| format!("{} exists", p.display())).unwrap_or_else(|| "(no QX_DS_YIELD_PATH)".into()),
            me.low as f64 / 1073741824.0,
            hold.as_secs_f64()
        );
        Some(me)
    }

    /// Between steps and while idle; looks at the host five times a second at most.
    fn check(&mut self, eng: &mut DsEngine) {
        if self.checked.elapsed() < std::time::Duration::from_millis(200) {
            return;
        }
        self.checked = std::time::Instant::now();
        let present = self.path.is_some() && qx_engine::dsv41::yield_path_alive();
        let avail = mem_available().unwrap_or(u64::MAX);
        let g = |b: u64| b as f64 / 1073741824.0;
        if let Some((when, before)) = self.released {
            if when.elapsed() >= std::time::Duration::from_secs(2) {
                self.released = None;
                // (the other side may have taken some of it already: a host that got none of it is the news; and a
                // part taken again within the two seconds says nothing)
                if !self.out {
                } else if avail < before + self.bytes / 2 && !present {
                    eprintln!("ram tier: WARNING: {:.1} GiB are available 2 s after the release, {:.1} before it", g(avail), g(before));
                } else {
                    eprintln!("ram tier: {:.1} GiB are available 2 s after the release ({:.1} before it)", g(avail), g(before));
                }
            }
        }
        if !self.out {
            if self.refilling && eng.ram_yield_stats().2 == 0 {
                self.refilling = false;
                eprintln!("ram tier: every record that had gone to NVMe is back in RAM");
            }
            if present || avail < self.low {
                let t0 = std::time::Instant::now();
                self.out = eng.ram_yield(true);
                let (dropped, moved, _) = eng.ram_yield_stats();
                eprintln!(
                    "ram tier: {:.1} GiB handed back to the host in {:.0} ms ({}; {:.1} GiB were available): {dropped} records now on NVMe, {moved} moved",
                    g(self.bytes),
                    1e3 * t0.elapsed().as_secs_f64(),
                    if present { "the yield path appeared" } else { "available memory under the low mark" },
                    g(avail)
                );
                self.roomy_since = None;
                self.refilling = false;
                self.released = Some((std::time::Instant::now(), avail));
            }
        } else if !present && avail > self.bytes + self.high {
            let since = *self.roomy_since.get_or_insert(self.checked);
            if since.elapsed() >= self.hold {
                let t0 = std::time::Instant::now();
                self.out = eng.ram_yield(false);
                self.roomy_since = None;
                if !self.out {
                    self.refilling = true;
                    eprintln!("ram tier: {:.1} GiB taken again in {:.0} ms ({:.1} GiB were available); {} records to read back", g(self.bytes), 1e3 * t0.elapsed().as_secs_f64(), g(avail), eng.ram_yield_stats().2);
                }
            }
        } else {
            self.roomy_since = None;
        }
    }
}

pub fn start(_g: &'static Gguf, tok: &'static Tok, opt: Options) -> Result<mpsc::Sender<Job>> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let pack: &'static Gguf = Box::leak(Box::new(Gguf::open(&opt.pack).map_err(|e| anyhow!("pack {}: {e}", opt.pack.display()))?));
    // A panic of the engine thread leaves a server that accepts requests and answers none (seen 2026-11-13: a
    // start-up graph capture failed and the process stayed up): leave instead, so a supervisor restarts it. In the
    // panic hook, before any unwinding: dropping the engine joins the driver thread, which may never return when
    // the panic is about the driver (`_exit`: no exit handlers under CUDA, no core image of the pinned tiers).
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        if std::thread::current().name() == Some(ENGINE_THREAD) {
            eprintln!("fatal: the engine thread panicked; exiting");
            unsafe { libc::_exit(101) };
        }
    }));
    std::thread::Builder::new().name(ENGINE_THREAD.into()).spawn(move || {
        // the engine thread off the pool's cores before this thread touches CUDA (rayon built wide first);
        // `dsv41::place_threads` tidies the rest inside the tiered load
        qx_engine::dsv41::pin_engine_early();
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
            // Every width the scheduler below can produce: the largest entry of WIDTHS at or under the decoders
            // plus the prompt tokens on offer, which several prompts arriving together take up to DS_TMAX. Every
            // one of them must compile and capture now (about 100 MB of device memory per graph): a kernel first
            // needed later may find no device memory left for its module once the graphs are instantiated
            // (2026-11-06: width 6, qx_mgemv_t6, failed to load with CUDA_ERROR_OUT_OF_MEMORY on every launch and
            // the windows ran without it), and `seal()` below turns any later compile into an abort.
            let widths: Vec<usize> = WIDTHS.iter().copied().filter(|&w| w <= DS_TMAX).collect();
            let toks = vec![0u32; 2 * DS_TMAX];
            eng.reset(0);
            let _ = eng.forward_window(&toks[..2], None); // positions 0-1: the stores' first rows, before any capture
            for &w in &widths {
                let _ = eng.forward_window(&toks[..w], None);
            }
            eng.reset(0);
            // the chunk path's kernels too (every format's expert kernels, the waits and marks), before the set seals
        if eng.chunk_rows() > 0 {
            eng.chunk_warm();  // takes the arena's flex tail for its dry pass
            eng.chunk_release();
        }
        qx_gpu::kernel::seal();
            let (free, total) = qx_gpu::mem::mem_info();
            eprintln!(
                "quixiinfer: warm-up {:.1}s (graphs for windows of {widths:?}); device memory free after warm-up: {:.2} GiB of {:.2}",
                t0.elapsed().as_secs_f64(),
                free as f64 / 2f64.powi(30),
                total as f64 / 2f64.powi(30)
            );
        }
        let mut width_hist: std::collections::BTreeMap<usize, usize> = Default::default();
        let prefix_on = !std::env::var("QX_DS_PREFIX").is_ok_and(|v| v == "0");
        let prefix_copy = !std::env::var("QX_DS_PREFIX_COPY").is_ok_and(|v| v == "0");
        let replay_cost: usize = std::env::var("QX_DS_REPLAY_COST").ok().and_then(|v| v.parse().ok()).unwrap_or(6_500);
        let mut prefix_copies = 0usize;
        let rewind_deep_min: usize = std::env::var("QX_DS_REWIND_DEEP_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(8_192);
        let skip_on = !std::env::var("QX_DS_SKIP").is_ok_and(|v| v == "0");
        let keep = eng.prompt_keep();
        let mut kv_only_windows = 0usize;
        let mut chunk_steps = 0usize;
        let mut multi_chunks = 0usize;
        let chunk_multi = !std::env::var("QX_DS_CHUNK_MULTI").is_ok_and(|v| v == "0");
        let chunk_f16_min: usize = std::env::var("QX_DS_CHUNK_F16_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        let chunk_beside = !std::env::var("QX_DS_CHUNK_BESIDE").is_ok_and(|v| v == "0");
        let panic_after: Option<usize> = std::env::var("QX_DS_PANIC_AFTER_CHUNKS").ok().and_then(|v| v.parse().ok());
        // The shortest prompt stretch that runs in chunks (QX_DS_CHUNK_MIN; 64 since 2026-11-13, max(window, 256)
        // before). Fresh three-slot 256k server, windows -> chunks: 79 tokens 3.6 -> 1.8 s, 128: 7.3 -> 1.8,
        // 186: 9.6 -> 2.2, 232: 9.5 -> 2.4. At 32 the 42-46-token loadgen prompts chunk too and lose (first
        // token 0.92 -> 1.22 s alone, three at once 13.3 -> 10.2 tok/s per request).
        let chunk_min: usize = std::env::var("QX_DS_CHUNK_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(64).max(1);
        let chunk_gap: usize = std::env::var("QX_DS_CHUNK_GAP").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        // window steps since the last chunk
        let mut gap_steps = usize::MAX / 2;
        let verbose_post = std::env::var("QX_DS_VERBOSE").is_ok();
        let mut post_left = 0usize;
        let mut post_t = std::time::Instant::now();
        let stats = std::env::var("QX_SCHED_STATS").is_ok();
        let group = eng.cfg.ratio.iter().copied().max().unwrap_or(1).max(1) as usize;
        // Prompt-lookup speculation (QX_DS_SPEC=K, the most drafts per request and step, default 15; 0: off): a
        // decoder whose last QX_DS_SPEC_N tokens (default 4) occurred before in its slot feeds what followed them
        // after its sample, and the step's rows are sampled in turn while the samples equal the drafts. Exact: a
        // window's rows are those of single steps, and after a rejected draft the slot is rewound (free: the
        // rings hold it) to the group boundary below and the tokens from there are fed again.
        let spec_max: usize = std::env::var("QX_DS_SPEC").ok().and_then(|v| v.parse().ok()).unwrap_or(15).min(DS_TMAX / 2);
        let spec_n: usize = std::env::var("QX_DS_SPEC_N").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(2);
        // Beside other decoders a draft row costs about three quarters of a token's share of the step, not half:
        // the match must be QX_DS_SPEC_N_MULTI tokens long (default 8) and three quarters of the drafts accepted.
        let spec_n_multi: usize = std::env::var("QX_DS_SPEC_N_MULTI").ok().and_then(|v| v.parse().ok()).unwrap_or(8).max(2);
        // (and at most QX_DS_SPEC_MULTI drafts, default 0, none: the other decoders get one token from the step however
        // wide it is. Measured 2026-11-18 with 3: a file edit beside an essay 30.9 -> 25.4 s, the essay 23.2 -> 29.4 s)
        let spec_max_multi: usize = std::env::var("QX_DS_SPEC_MULTI").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(spec_max);
        // Suffix decoding (vLLM's): the replies of finished requests, up to QX_DS_SPEC_CORPUS tokens of them (default
        // 262,144; 0: none), are a second source of drafts. A draft is only ever a guess the model's own rows
        // confirm, so no reply's content reaches another request; its speed can (a reply drafted from another
        // request's is faster): 0 for a server whose clients must not learn that. Measured 2026-11-19 with 11,700
        // tokens of 39 replies in it: three fresh tasks 34.62 -> 35.41 s (+2.3%), a reply asked again -41 to -48%.
        let corpus_max: usize = std::env::var("QX_DS_SPEC_CORPUS").ok().and_then(|v| v.parse().ok()).unwrap_or(262_144);
        // (a place in another reply must agree over QX_DS_SPEC_N_CORPUS tokens, default 8: a full corpus of replies
        // in one style agrees over 4 with almost anything)
        let spec_n_corpus: usize = std::env::var("QX_DS_SPEC_N_CORPUS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let mut corpus: Vec<u32> = Vec::new();
        let mut spec_corpus_steps = 0usize;
        let mut spec_trips = 0usize;
        let mut n_dec = 0usize;
        let mut spec_sat_out = 0usize;
        // (QX_DS_SPEC_COOL: steps a source sits out after a step with no draft accepted, and the first time its
        // drafts have cost more than they gave, default 8; twice as long each time in a row, 256 at most)
        let spec_cool: usize = std::env::var("QX_DS_SPEC_COOL").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let admit_window: usize = std::env::var("QX_DS_ADMIT_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let (mut front_skips, mut admit_reordered) = (0usize, 0usize);
        let (mut spec_steps, mut spec_drafted, mut spec_accepted, mut spec_refed) = (0usize, 0usize, 0usize, 0usize);
        let mut last_save: Option<std::time::Instant> = None;
        let mut free: Vec<usize> = (0..eng.slots).rev().collect();
        let mut waiting: std::collections::VecDeque<Job> = Default::default();
        let mut running: Vec<Req> = Vec::new();
        let mut steps = 0usize;
        // per-step timing for the `sched:` line (QX_SCHED_STATS): the engine's forward time and its phases, the
        // Engram gather, and the wall time between forwards (sampling, detokenizing, the HTTP side)
        let mut acc_fwd = 0f64;
        let mut acc_phase = [0f64; 5];
        let mut acc_engram = 0f64;
        let mut acc_wall = 0f64;
        let mut acc_steps = 0usize;
        let mut last_step_end: Option<std::time::Instant> = None;
        let mut yielder = Yielder::new(&eng);
        loop {
            if running.is_empty() && waiting.is_empty() {
                match &mut yielder {
                    // idle: the host is still looked at
                    Some(y) => loop {
                        match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                            Ok(j) => break waiting.push_back(j),
                            Err(mpsc::RecvTimeoutError::Timeout) => y.check(&mut eng),
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                    },
                    None => match rx.recv() {
                        Ok(j) => waiting.push_back(j),
                        Err(_) => return,
                    },
                }
            }
            if let Some(y) = &mut yielder {
                y.check(&mut eng);
            }
            while let Ok(j) = rx.try_recv() {
                waiting.push_back(j);
            }
            // admit: a free slot each, the one sharing the longest prefix with the prompt when there is one
            while !free.is_empty() {
                // Among the first QX_DS_ADMIT_WINDOW waiting requests (default 4; 1: arrival order), the one that
                // shares the longest prefix (256+ tokens) with a free slot goes first: its prompt work is saved
                // before another request resets that slot. The front request is passed over three times at most.
                let mut wi = 0;
                if prefix_on && admit_window > 1 && waiting.len() > 1 && front_skips < 3 {
                    let mut best = 255;
                    for (i, j) in waiting.iter().enumerate().take(admit_window) {
                        // (what the admission below would reuse: a prefix the rings reach, or a deep one)
                        let l = free.iter().map(|&sl| {
                            let mut l = eng.tokens(sl).iter().zip(&j.prompt).take_while(|(a, b)| a == b).count().min(j.prompt.len().saturating_sub(1));
                            l -= l % group;
                            if l >= 256 && eng.rewind_fast(sl, l) { l } else if l >= rewind_deep_min { l.saturating_sub(replay_cost) } else { 0 }
                        }).max().unwrap_or(0);
                        if l > best {
                            best = l;
                            wi = i;
                        }
                    }
                }
                front_skips = if wi == 0 { 0 } else { front_skips + 1 };
                admit_reordered += (wi != 0) as usize;
                let Some(job) = waiting.remove(wi) else { break };
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
                // The prefix of the prompt slot `slot` holds (to a group boundary), and what reusing it is worth in
                // tokens of prompt work saved. A prefix whose end lies within the slot's last window, with every
                // layer's ring exact there (`rewind_fast`), costs nothing to return to. One that ends earlier needs
                // the rings rebuilt by a replay of up to `replay_span` (5,208) tokens in one trimmed chunk, ~4.4 s,
                // the price of ~6,500 tokens computed afresh (`QX_DS_REPLAY_COST`): it is reused only from
                // QX_DS_REWIND_DEEP_MIN tokens (default 8,192) and counts for its length less that price. Until
                // 2026-11-14 every prefix of 256+ tokens was reused and the replay ran in windows: 2,047 tokens with
                // 1,252 reused 28.5 s at the client against 5.2 s afresh.
                let shared = |slot: usize| -> (usize, usize) {
                    let mut l = eng.tokens(slot).iter().zip(&job.prompt).take_while(|(a, b)| a == b).count().min(job.prompt.len() - 1);
                    l -= l % group;
                    if l < 256 {
                        (0, 0)
                    } else if eng.rewind_fast(slot, l) {
                        (l, l)
                    } else if l >= rewind_deep_min {
                        (l, l.saturating_sub(replay_cost).max(1))
                    } else {
                        (0, 0)
                    }
                };
                // (among free slots worth the same, the one holding the fewest tokens: a prompt that reuses nothing
                // takes an empty slot before it resets a cached conversation, and the shortest one before a longer.
                // Until 2026-11-14 ties went to the last free slot in the list: a 145-token prompt reset the slot
                // holding a 12.8k-token prompt while two slots were empty)
                let (i, reuse, worth) = if prefix_on {
                    free.iter().enumerate().map(|(i, &sl)| { let (l, v) = shared(sl); (i, l, v) }).max_by_key(|&(i, _, v)| (v, std::cmp::Reverse(eng.tokens(free[i]).len()))).unwrap()
                } else {
                    (free.len() - 1, 0, 0)
                };
                // Prefix reuse across slots (QX_DS_PREFIX_COPY=0: off): when another slot, a running one, holds a
                // prefix of this prompt worth 256+ tokens more than the chosen free slot's, its stores, rings and
                // tokens are copied into the chosen slot (which holds nothing of this prompt, or a shorter prefix of
                // it), and that slot is rewound there. The source goes on undisturbed. A source still inside its
                // prompt has no exact rings above the last KV source (`rewind_fast` is false there): it is copied
                // only as a deep prefix, with a replay.
                let mut copy_from: Option<(usize, usize)> = None;
                if prefix_on && prefix_copy {
                    let mut best = worth + 255;
                    for s in (0..opt.slots).filter(|s| !free.contains(s)) {
                        let (l, v) = shared(s);
                        if v > best {
                            best = v;
                            copy_from = Some((s, l));
                        }
                    }
                }
                let reuse = copy_from.map_or(reuse, |(_, l)| l);
                let slot = free.remove(i);
                // (the request's clock starts before the rewind: a deep one replays up to 5,208 tokens, and until
                // 2026-11-14 the prompt time left that out)
                let t_admit = std::time::Instant::now();
                if let Some((s, l)) = copy_from {
                    eng.copy_prefix(s, slot, l);
                    prefix_copies += 1;
                } else if reuse > 0 {
                    eng.rewind(slot, reuse);
                } else {
                    eng.reset(slot);
                }
                let job_think = job.grammar_after_think;
                running.push(Req { job, slot, detok: Detok { tok, ids: Vec::new(), sent: 0 }, text: String::new(), n: 0, t0: t_admit, t_prompt: None, finish: None, in_think: job_think && matcher.is_some(), matcher, read: reuse, reused: reuse, next: None, logits: None, refeed: Vec::new(), drafts: Vec::new(), srow: None, spec: [SPEC_NEW; 4], spec_src: 0 });
            }
            if running.is_empty() {
                continue;
            }
            let mut segs: Vec<(usize, u32)> = Vec::new();
            let mut plan: Vec<(usize, usize, bool)> = Vec::new(); // (request index, tokens, completes the prompt or decodes)
            let mut out: Vec<f32> = Vec::new();
            // the chunk path (QX_DS_CHUNK=N at start): the longest remaining prompt stretch of at least
            // `chunk_min` tokens goes through forward_chunk, up to N rows at once, KV-only when at least
            // prompt_keep positions follow. (Since 2026-11-10 a chunk's dense products are f16 tensor-core GEMMs,
            // KL-bounded against the exact path, while windows stay exact; until 2026-11-13 a prompt that arrived
            // beside a decoder ran in exact windows, so its temperature-0 output depended on that.
            // QX_DS_PREFILL_EXACT=1 keeps chunks exact; chunks under 256 rows are exact anyway: QX_DS_CHUNK_F16_MIN.)
            let chunk_rows = eng.chunk_rows();
            // Beside decoders (since 2026-11-13; QX_DS_CHUNK_BESIDE=0: chunks only while nobody decodes): such a
            // stretch still runs in chunks, and the decoders get `chunk_gap` steps of their own between two chunks
            // (QX_DS_CHUNK_GAP, default 4). A decoder waits a chunk (4-6 s), but a 128-wide full-stack window beside
            // it took 2.5 s: a 10,031-token prompt arriving beside a reply 196.6 -> 9.1 s, the reply 197 -> 18.9 s.
            let decoding = running.iter().any(|r| r.next.is_some());
            let chunk_due = !decoding || (chunk_beside && gap_steps >= chunk_gap);
            let chunked = if chunk_rows > 0 && chunk_due {
                let pick = (0..running.len()).filter(|&i| running[i].job.prompt.len() - running[i].read >= chunk_min).max_by_key(|&i| running[i].job.prompt.len() - running[i].read);
                if let Some(ri) = pick {
                    let r = &running[ri];
                    let rem = r.job.prompt.len() - r.read;
                    // (the last prompt_keep positions alone need the full stack: a chunk's rows before them stop at
                    // the last KV source. QX_DS_CHUNK_MIXED=0: a KV-only chunk up to them, then a chunk of their own)
                    let mixed = !std::env::var("QX_DS_CHUNK_MIXED").is_ok_and(|v| v == "0");
                    // (a prompt longer than a chunk: KV-only chunks up to the kept rows, so the full stack, whose
                    // experts take about 6 s to stream however few the rows, runs in the last chunk only)
                    let n = if skip_on && rem > keep && (!mixed || rem > chunk_rows) { rem.min(chunk_rows).min(rem - keep) } else { rem.min(chunk_rows) };
                    let keep_from = if skip_on { (rem.saturating_sub(keep)).min(n) } else { 0 };
                    // Several short stretches in one chunk (QX_DS_CHUNK_MULTI=0: one request per chunk): the other
                    // running requests whose remaining prompt stretch would run in a chunk of its own, exact like
                    // this one (each under the f16 threshold: a part's values are those of its own chunk, so a reply
                    // does not depend on its company), while the rows fit. The layer's experts are staged once.
                    let mut mates: Vec<usize> = Vec::new();
                    if chunk_multi && keep_from == 0 && n == rem && rem < chunk_f16_min {
                        let mut total = n;
                        for i in 0..running.len() {
                            let q = &running[i];
                            let m = q.job.prompt.len() - q.read;
                            if i != ri && q.next.is_none() && m >= chunk_min && m < chunk_f16_min && !(skip_on && m > keep) && total + m <= chunk_rows {
                                mates.push(i);
                                total += m;
                            }
                        }
                    }
                    let mut first_out: Option<Vec<f32>> = None;
                    if !mates.is_empty() {
                        let mut idx = vec![ri];
                        idx.extend(&mates);
                        let mut outs = {
                            let parts: Vec<(usize, &[u32])> = idx.iter().map(|&i| (running[i].slot, &running[i].job.prompt[running[i].read..])).collect();
                            eng.forward_chunk_parts(&parts)
                        };
                        first_out = Some(std::mem::take(&mut outs[0]));
                        for (&i, o) in idx.iter().zip(outs).skip(1) {
                            let q = &mut running[i];
                            q.read = q.job.prompt.len();
                            q.t_prompt = Some(q.t0.elapsed().as_secs_f64());
                            q.logits = Some(o);
                        }
                        multi_chunks += 1;
                    }
                    // whether another running request still has a prompt stretch that would run in chunks
                    let others_pending = (0..running.len()).filter(|&i| i != ri).any(|i| running[i].job.prompt.len() - running[i].read >= chunk_min);
                    let r = &mut running[ri];
                    let last = match first_out { Some(o) => o, None => eng.forward_chunk_keep(r.slot, &r.job.prompt[r.read..r.read + n], keep_from) };
                    r.read += n;
                    // the flex tail back to the tiers once this prompt's chunks are done and no other running request
                    // still has a prompt stretch that would run in chunks
                    if r.read == r.job.prompt.len() && !others_pending {
                        let tr = std::time::Instant::now();
                        eng.chunk_release();
                        if verbose_post {
                            eprintln!("post-prompt: flex tail released in {:.1} ms", tr.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                    chunk_steps += 1;
                    gap_steps = 0;
                    // (a test of the panic hook above: QX_DS_PANIC_AFTER_CHUNKS=N panics here after the Nth chunk)
                    if panic_after == Some(chunk_steps) {
                        panic!("QX_DS_PANIC_AFTER_CHUNKS={chunk_steps}");
                    }
                    if r.read == r.job.prompt.len() {
                        r.t_prompt = Some(r.t0.elapsed().as_secs_f64());
                        r.logits = Some(last);
                        post_left = 12;
                        post_t = std::time::Instant::now();
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !chunked {
            // the window: decoders first (one token each), then prompt tokens up to the largest width of WIDTHS
            // the prompts can fill (at most `window` per prompt)
            // (with speculation: the tokens to feed again, the sample, then the drafts; drafts only in a step
            // without prompt tokens, and as many as bring the step to a width of WIDTHS)
            let prompt_rows = running.iter().any(|r| r.next.is_none() && r.read < r.job.prompt.len());
            // (tokens fed again can bring the decoders' rows to a width the server never loaded, 5 or 7: decoders
            // with such tokens then sit this step out, last first, until the rows are a width of WIDTHS)
            let mut out_now: Vec<bool> = vec![false; running.len()];
            let mut base: usize = running.iter().filter(|r| r.next.is_some()).map(|r| r.refeed.len() + 1).sum();
            if running.iter().any(|r| r.next.is_some() && !r.refeed.is_empty()) {
                for ri in (0..running.len()).rev() {
                    if WIDTHS.contains(&base) {
                        break;
                    }
                    if running[ri].next.is_some() && !running[ri].refeed.is_empty() {
                        out_now[ri] = true;
                        base -= running[ri].refeed.len() + 1;
                        spec_sat_out += 1;
                    }
                }
            }
            let mut wanted: Vec<Vec<u32>> = vec![Vec::new(); running.len()];
            n_dec = running.iter().filter(|r| r.next.is_some()).count();
            if spec_max > 0 && !prompt_rows {
                for (ri, r) in running.iter_mut().enumerate() {
                    let Some(id) = r.next else { continue };
                    if out_now[ri] {
                        continue;
                    }
                    // (each kind sits its own steps out)
                    let ready = [r.spec[0].cool == 0, r.spec[1].cool == 0, r.spec[2].cool == 0, r.spec[3].cool == 0];
                    for sp in r.spec.iter_mut() {
                        sp.cool = sp.cool.saturating_sub(1);
                    }
                    let pos = eng.pos(r.slot);
                    // (not under a constraint: its mask follows the samples one by one; not before the rings
                    // can take the rewind; not where a rewind would cross the prompt's end and its checkpoint;
                    // not past the request's or the context's end)
                    let left = r.job.max_tokens.saturating_sub(r.n + 1).min(opt.max_ctx.saturating_sub(pos + r.refeed.len() + 3));
                    if r.matcher.is_some() || pos < r.job.prompt.len() + group || left == 0 || !eng.rewind_fast(r.slot, pos - pos % group) {
                        continue;
                    }
                    let mut hist = eng.tokens(r.slot).to_vec();
                    hist.extend(&r.refeed);
                    // (the end of the sequence: the last 23 tokens and the sample)
                    let mut suf = hist[hist.len().saturating_sub(23)..].to_vec();
                    suf.push(id);
                    let n = if n_dec > 1 { spec_n_multi } else { spec_n };
                    let cap = (if n_dec > 1 { spec_max_multi } else { spec_max }).min(left);
                    // its own slot's history, or the finished replies of every request where they agree longer;
                    // a short agreement earns a short draft (half the agreed length), a long one its kind's length
                    let (m_own, own) = suffix_draft(&hist, &suf, n, cap);
                    let (m_corp, corp) = if corpus.is_empty() { (0, Vec::new()) } else { suffix_draft(&corpus, &suf, n.max(spec_n_corpus), cap) };
                    let kind = |src: usize, m: usize| 2 * src + (m >= 16) as usize;
                    let (ko, kc) = (kind(0, m_own), kind(1, m_corp));
                    let own_ok = !own.is_empty() && ready[ko];
                    let corp_ok = !corp.is_empty() && ready[kc];
                    let (kd, m, mut d) = if corp_ok && (m_corp > m_own || !own_ok) { (kc, m_corp, corp) } else if own_ok { (ko, m_own, own) } else { continue };
                    d.truncate(r.spec[kd].k.min(if m < 16 { (m / 2).max(1) } else { usize::MAX }));
                    if kd >= 2 {
                        spec_corpus_steps += 1;
                    }
                    r.spec_src = kd;
                    wanted[ri] = d;
                }
                let all: usize = wanted.iter().map(|w| w.len()).sum();
                let mut give = WIDTHS.iter().copied().filter(|&w| w >= base && w <= base + all && w <= DS_TMAX).max().map_or(0, |w| w - base);
                for w in wanted.iter_mut() {
                    w.truncate(give);
                    give -= w.len();
                }
            }
            for (ri, r) in running.iter_mut().enumerate() {
                if out_now[ri] {
                    continue;
                }
                if let Some(id) = r.next {
                    r.srow = Some(segs.len() + r.refeed.len());
                    segs.extend(r.refeed.iter().map(|&t| (r.slot, t)));
                    segs.push((r.slot, id));
                    r.drafts = std::mem::take(&mut wanted[ri]);
                    segs.extend(r.drafts.iter().map(|&t| (r.slot, t)));
                    plan.push((ri, r.refeed.len() + 1 + r.drafts.len(), true));
                    spec_refed += r.refeed.len();
                    if !r.drafts.is_empty() {
                        spec_steps += 1;
                        spec_drafted += r.drafts.len();
                    }
                    r.refeed.clear();
                }
            }
            let d = segs.len();
            gap_steps += 1;
            // a prompt stretch waiting for its chunk (beside decoders) stays out of the windows between chunks
            let held = |r: &Req| chunk_rows > 0 && chunk_beside && d > 0 && r.job.prompt.len() - r.read >= chunk_min;
            let avail: usize = running.iter().filter(|r| r.next.is_none() && r.read < r.job.prompt.len() && !held(r)).map(|r| (r.job.prompt.len() - r.read).min(window)).sum();
            let need = (d + avail).min(DS_TMAX);
            let width = WIDTHS.iter().copied().filter(|&w| w <= need && w <= DS_TMAX).max().unwrap_or(d).max(d);
            let mut room = width - d;
            for (ri, r) in running.iter().enumerate() {
                if room == 0 {
                    break;
                }
                if r.next.is_none() && r.read < r.job.prompt.len() && !held(r) {
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
            // a window of prompt chunks only, each at least prompt_keep() positions before its prompt's end, runs
            // KV-only (QX_DS_SKIP=0 disables): exact for every kept position, about half the work
            let kv_only = skip_on && d == 0 && plan.iter().all(|&(ri, n, _)| running[ri].job.prompt.len() - (running[ri].read + n) >= keep);
            if kv_only {
                kv_only_windows += 1;
            }
            out = eng.forward_segments_opt(&segs, None, kv_only);
            if verbose_post && post_left > 0 {
                // QX_DS_VERBOSE: the first steps after a prompt's last chunk (the forward's time and the wall time
                // since the previous step or the chunk's end)
                eprintln!("post-prompt step {}: forward {:.1} ms, {:.1} ms since the previous one ended (width {}; uploads {:.2}, launch {:.2}, wait {:.2}, flush {:.2}, download {:.2}){}", 12 - post_left, eng.t_forward * 1e3, post_t.elapsed().as_secs_f64() * 1e3, segs.len(), eng.t_phase[0] * 1e3, eng.t_phase[1] * 1e3, eng.t_phase[2] * 1e3, eng.t_phase[3] * 1e3, eng.t_phase[4] * 1e3, eng.tier_stats().unwrap_or_default());
                post_left -= 1;
                post_t = std::time::Instant::now();
            }
            }
            steps += 1;
            if stats {
                let now = std::time::Instant::now();
                if let Some(t) = last_step_end {
                    // the step period less the engine's own time: what the host spent outside the forward
                    acc_wall += (now.duration_since(t).as_secs_f64() - eng.t_forward).max(0.0);
                }
                acc_fwd += eng.t_forward;
                acc_engram += eng.t_engram;
                for (a, v) in acc_phase.iter_mut().zip(eng.t_phase.iter()) {
                    *a += v;
                }
                acc_steps += 1;
                last_step_end = Some(now);
            }
            let mut row = 0;
            let mut prompt_done = false;
            for &(ri, n, samples) in &plan {
                let r = &mut running[ri];
                if r.next.is_none() {
                    r.read += n;
                    if r.read == r.job.prompt.len() {
                        r.t_prompt = Some(r.t0.elapsed().as_secs_f64());
                        prompt_done = true;
                    }
                }
                if samples && r.srow.is_none() {
                    r.logits = Some(out[(row + n - 1) * n_vocab..(row + n) * n_vocab].to_vec());
                }
                row += n;
            }
            // a prompt that ended in a window (a remainder under `chunk_min` after its chunks: QX_DS_SKIP=0 or a small
            // QX_DS_CHUNK) gives the flex tail back too, once no stretch waits for a chunk (a no-op when not held)
            if prompt_done && chunk_rows > 0 && !running.iter().any(|r| r.next.is_none() && r.job.prompt.len() - r.read >= chunk_min) {
                eng.chunk_release();
            }
            // sample for every request with a distribution
            for r in running.iter_mut() {
                // a decoder's rows: its sample's, then one per draft, read while the samples equal the drafts
                let srow = r.srow.take();
                if srow.is_none() && r.logits.is_none() {
                    continue;
                }
                let drafts = std::mem::take(&mut r.drafts);
                let k = drafts.len();
                let pos_first = eng.pos(r.slot) - k;
                let mut accepted = 0usize;
                for i in 0..=k {
                let mut logits = match srow {
                    Some(b) => out[(b + i) * n_vocab..(b + i + 1) * n_vocab].to_vec(),
                    None => r.logits.take().unwrap(),
                };
                let pos = pos_first + i;
                if pos == r.job.prompt.len() && pos >= 256 {
                    // the prompt's end: the rings kept, so the next turn's rewind to here is free however long
                    // the reply (a client that drops the reasoning from the history shares exactly this prefix)
                    eng.checkpoint(r.slot);
                }
                if let Some(m) = r.matcher.as_mut().filter(|_| !r.in_think) {
                    match m.compute_mask() {
                        Ok(mask) => apply_mask(&mut logits, mask.as_slice()),
                        Err(e) => {
                            let _ = r.job.events.send(Event::Error(format!("grammar: {e}")));
                            r.finish = Some("error");
                            break;
                        }
                    }
                }
                let id = sample(&logits, &r.job.sampling, pos);
                let was_think = r.in_think;
                if let Some(m) = r.matcher.as_mut().filter(|_| !was_think) {
                    if !stop_ids.contains(&id) {
                        if let Err(e) = m.consume_token(id) {
                            let _ = r.job.events.send(Event::Error(format!("grammar: {e}")));
                            r.finish = Some("error");
                            break;
                        }
                    }
                }
                let live = r.emit(id, &stop_ids, opt.max_ctx, pos);
                if r.in_think && r.text.contains("</think>") {
                    r.in_think = false;
                }
                if !live {
                    break;
                }
                if r.matcher.as_ref().is_some_and(|m| m.is_stopped()) {
                    r.finish = Some("stop");
                    break;
                }
                if i < k && id == drafts[i] {
                    accepted += 1;
                    continue;
                }
                r.next = Some(id);
                break;
                }
                if k > 0 {
                    spec_accepted += accepted;
                    // What the step was worth to its source, in draft rows; a source whose drafts have cost more than
                    // they gave lately sits out, twice as long each time in a row (QX_DS_SPEC_COOL steps the first,
                    // 256 at most), and starts again from short drafts. A list of another topic agrees in its
                    // structure and not in its values: until 2026-11-19 one rate for both sources (under half
                    // accepted: 16 steps out) let such drafts cost a first-time JSON list 4-5%.
                    let end = pos_first + accepted;
                    let fed_again = if accepted < k { end % group } else { 0 };
                    // (a rejection also costs the rewind and the experts its rows fetched for nothing: counted as two
                    // rows; with one, a list that agrees long with another and fails at each key still read +3.5%)
                    let net = (if n_dec > 1 { 1.33 } else { 2.0 }) * accepted as f32 - k as f32 - fed_again as f32 - if accepted < k { 2.0 } else { 0.0 };
                    let sp = &mut r.spec[r.spec_src];
                    sp.gain = 0.7 * sp.gain + 0.3 * net;
                    if accepted == k {
                        // (the next length from the one that was accepted, not from one a short agreement cut)
                        sp.k = sp.k.max((2 * k + 1).min(spec_max));
                    } else {
                        sp.k = accepted.max(1);
                        if accepted == 0 {
                            sp.cool = sp.cool.max(spec_cool);
                        }
                    }
                    // (the doublings are forgotten only by drafts that pay well: a short draft's best is 2, so a kind
                    // that has cost more than it gave stays mostly out for the request; a copy, of the long kind,
                    // has its own record and is found again once 16 of its tokens have been written)
                    if sp.gain > 2.0 {
                        sp.trips = 0;
                    }
                    if sp.gain < 0.0 {
                        sp.cool = (spec_cool << sp.trips.min(5)).min(256);
                        sp.trips += 1;
                        sp.gain = 0.5;
                        sp.k = 2;
                        spec_trips += 1;
                    }
                    if accepted < k {
                        if r.finish.is_none() {
                            // the slot ran the rejected drafts: back to the group boundary under the accepted
                            // tokens' end, and what lies between is fed again with the next sample
                            let end = pos_first + accepted;
                            let g = end - end % group;
                            r.refeed = eng.tokens(r.slot)[g..end].to_vec();
                            eng.rewind(r.slot, g);
                        }
                    }
                }
            }
            // finished requests give their slots back (their tokens stay for a prefix match)
            let mut i = 0;
            while i < running.len() {
                if running[i].finish.is_some() {
                    let r = running.remove(i);
                    if spec_max > 0 && corpus_max > 0 {
                        // the reply's tokens as the slot holds them (the last sample was never fed)
                        let toks = eng.tokens(r.slot);
                        let (a, b) = (r.job.prompt.len().min(toks.len()), (r.job.prompt.len() + r.n).min(toks.len()));
                        if b > a + 8 {
                            corpus.extend_from_slice(&toks[a..b]);
                            corpus.push(SEP);
                            if corpus.len() > corpus_max {
                                let cut = corpus.len() - corpus_max;
                                corpus.drain(..cut);
                            }
                        }
                    }
                    r.done(stats, &eng.tier_stats().unwrap_or_default());
                    free.push(r.slot);
                } else {
                    i += 1;
                }
            }
            if stats && (steps % 200 == 0 || running.is_empty()) {
                eprintln!("sched: {steps} steps, {} running, {} waiting, window widths {:?}, KV-only windows {kv_only_windows}, chunk steps {chunk_steps} ({multi_chunks} of several slots), prefix copies {prefix_copies}, speculation {spec_steps} steps: {spec_accepted} of {spec_drafted} drafts accepted, {spec_refed} tokens fed again, {spec_corpus_steps} steps drafted from other replies, {spec_trips} sit-outs for drafts that cost more than they gave ({spec_sat_out} decoders sat a step out), {admit_reordered} admitted out of order", running.len(), waiting.len(), width_hist);
                if acc_steps > 0 {
                    let k = acc_steps as f64 / 1e3;
                    // the wall between forwards is everything outside the engine: host sampling, detokenizing,
                    // event sends and the scheduler; forward = the engine's own time per step
                    eprintln!(
                        "sched: per step over the last {acc_steps}: forward {:.2} ms (engram {:.2}, uploads {:.2}, launch/replay {:.2}, device wait {:.2}, history flush {:.2}, logits download {:.2}), outside the engine {:.2} ms",
                        acc_fwd / k, acc_engram / k, acc_phase[0] / k, acc_phase[1] / k, acc_phase[2] / k, acc_phase[3] / k, acc_phase[4] / k,
                        acc_wall / (acc_steps.saturating_sub(1).max(1) as f64 / 1e3)
                    );
                    acc_fwd = 0.0;
                    acc_engram = 0.0;
                    acc_phase = [0.0; 5];
                    acc_wall = 0.0;
                    acc_steps = 0;
                    last_step_end = None;
                }
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
    }).map_err(|e| anyhow!("engine thread: {e}"))?;
    ready_rx.recv().map_err(|_| anyhow!("engine thread ended"))?.map_err(|e| anyhow!(e))?;
    Ok(tx)
}
