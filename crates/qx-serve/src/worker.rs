//! The engine thread: continuous batching over the engine's sequence slots, tokens streamed back over a channel
//! as text deltas.
//!
//! Each loop admits waiting requests into free slots, then runs one step. A request alone (nothing else running
//! or waiting) decodes speculatively; as soon as another arrives it joins the others, and every running request
//! advances in one window per step: one token each, or up to `PROMPT_CHUNK` prompt tokens for one still reading its
//! prompt. Prompts long enough for the batched prefill are prefilled on admission (the others wait meanwhile).
use anyhow::Result;
use qx_engine::engine::{Engine, Sampling, SpecRun};
use qx_pack::gguf::Gguf;
use qx_tok::Tok;
use std::sync::mpsc;

pub struct Job {
    pub arrived: std::time::Instant,
    pub prompt: Vec<u32>,
    /// A structured-output constraint.
    pub grammar: Option<crate::grammar::Spec>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop: Vec<String>,
    pub events: tokio::sync::mpsc::UnboundedSender<Event>,
}

pub enum Event {
    /// Newly decoded text.
    Delta(String),
    Done { prompt_tokens: usize, completion_tokens: usize, finish: &'static str, prompt_tps: f64, decode_tps: f64 },
    Error(String),
}

/// Decodes a growing token list into text deltas, holding back bytes until they form whole UTF-8 characters.
struct Detok<'a> {
    tok: &'a Tok,
    ids: Vec<u32>,
    sent: usize,
}

impl<'a> Detok<'a> {
    fn push(&mut self, id: u32) -> Option<String> {
        self.ids.push(id);
        let text = self.tok.decode(&self.ids).ok()?;
        if text.ends_with('\u{fffd}') || text.len() <= self.sent {
            return None;
        }
        let d = text[self.sent..].to_string();
        self.sent = text.len();
        Some(d)
    }
}

/// Lanes of a window given first to requests reading their prompts (the rest decode; decoders that do not fit
/// skip the step, in turn), and the most one prompt takes per step.
const PROMPT_LANES: usize = 8;
const PROMPT_CHUNK: usize = 8;
/// A cached prefix shorter than this is not worth restoring (the restore copies about 110 MiB of state), and a
/// prompt shorter than this is not saved before its last token (the snapshot stalls the pipeline for a step).
const PREFIX_MIN: usize = 64;
const PREFIX_PROMPT_MIN: usize = 128;
/// From this many remaining prompt tokens on, a request reads its prompt with whole prefill chunks between steps:
/// a chunk stages every non-resident expert, which only a few hundred tokens amortize.
const PREFILL_MIN_BATCH: usize = 256;

/// A request holding a slot.
struct Running<'a> {
    job: Job,
    slot: usize,
    detok: Detok<'a>,
    text: String,
    /// Tokens passed on.
    n: usize,
    t0: std::time::Instant,
    t_prompt: Option<f64>,
    finish: Option<&'static str>,
    /// Present while the request decodes speculatively.
    spec: Option<SpecRun>,
    /// The prompt was saved to the prefix cache before its last token.
    prompt_saved: bool,
    /// The structured-output matcher, fed every generated token; and the window lane whose sample it masked.
    matcher: Option<llguidance::Matcher>,
    lane: Option<usize>,
}

impl Running<'_> {
    /// Passes a generated token on; false once the request is finished.
    fn emit(&mut self, id: u32, stop_ids: &[u32], max_ctx: usize, pos: usize) -> bool {
        if self.t_prompt.is_none() {
            self.t_prompt = Some(self.t0.elapsed().as_secs_f64());
        }
        if stop_ids.contains(&id) {
            self.finish = Some("stop");
            return false;
        }
        if let Some(delta) = self.detok.push(id) {
            self.text.push_str(&delta);
            if let Some(st) = self.job.stop.iter().find(|st| !st.is_empty() && self.text.contains(st.as_str())) {
                // send what precedes the stop string, then end
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

    fn done(&self) {
        let all = self.t0.elapsed().as_secs_f64();
        let tp = self.t_prompt.unwrap_or(all);
        if std::env::var("QX_SCHED_STATS").is_ok() {
            eprintln!("req: {} prompt tokens, queued {:.0} ms, first token {:.0} ms after admission, {} tokens in {:.2}s, {}", self.job.prompt.len(), 1e3 * (self.t0 - self.job.arrived).as_secs_f64(), 1e3 * tp, self.n, all, self.finish.unwrap_or("length"));
        }
        let _ = self.job.events.send(Event::Done {
            prompt_tokens: self.job.prompt.len(),
            completion_tokens: self.n,
            finish: self.finish.unwrap_or("length"),
            prompt_tps: (self.job.prompt.len() as f64 - 1.0).max(0.0) / tp.max(1e-9),
            decode_tps: self.n as f64 / (all - tp).max(1e-9),
        });
    }
}

/// How the engine thread shares the GPU between decode steps and prompt reading.
#[derive(Clone, Copy)]
pub struct Options {
    pub slots: usize,
    /// Time a prefill chunk may take between two decode steps (its size follows from the measured rate).
    pub prefill_budget_ms: f64,
}

pub fn start(g: &'static Gguf, tok: &'static Tok, max_ctx: usize, cache_budget: usize, order: Vec<(usize, usize)>, opt: Options) -> Result<mpsc::Sender<Job>> {
    let slots = opt.slots;
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        let mut eng = match Engine::load_slots(g, max_ctx, cache_budget, &order, slots) {
            Ok(mut e) => {
                e.warm_up();
                let _ = ready_tx.send(Ok(()));
                e
            }
            Err(e) => {
                let _ = ready_tx.send(Err(format!("{e:#}")));
                return;
            }
        };
        let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.specials.get(*s).copied()).collect();
        let grammars = match crate::grammar::Grammars::new(tok, eng.cfg.n_vocab) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("structured outputs unavailable: {e:#}");
                let _ = ready_tx.send(Err(format!("{e:#}")));
                return;
            }
        };
        let mut free: Vec<usize> = (0..eng.slots).rev().collect();
        let mut waiting: std::collections::VecDeque<Job> = Default::default();
        let mut running: Vec<Running> = Vec::new();
        // The step queued last and not yet read: (slot, position of its last token) per segment. Its samples
        // are read after the next step is queued, so the GPU never waits for the host; a request found finished
        // then has one more token in flight, which is dropped.
        let mut pending: Vec<(usize, usize)> = Vec::new();
        // QX_SCHED_STATS: every 200 steps, the mix of steps and their time
        let stats = std::env::var("QX_SCHED_STATS").is_ok();
        let r#async = !std::env::var("QX_ASYNC").is_ok_and(|v| v == "0");
        let prefix_on = !std::env::var("QX_PREFIX").is_ok_and(|v| v == "0");
        eng.snapshot_prefill = prefix_on;
        // a prompt keeps the cache's tail lent across its chunks; given back when no prompt is left
        eng.keep_lent = true;
        let (mut st_n, mut st_spec, mut st_tok, mut st_t, mut st_admit, mut st_wait) = (0usize, 0usize, 0usize, 0f64, 0f64, 0f64);
        let mut hist: std::collections::BTreeMap<usize, (usize, f64)> = Default::default();
        // prefill rate (tokens per second, running average) and chunk accounting
        let (mut pf_rate, mut pf_chunks, mut pf_tokens, mut pf_time) = (1000f64, 0usize, 0usize, 0f64);
        let pf_max = eng.prefill_chunk_tokens();
        // the decoder that took the last lane when not all fitted (usize::MAX: all did)
        let mut rr_last = usize::MAX;
        let mut c_last = (eng.cache_stats(), eng.shared().cpu_ns.load(std::sync::atomic::Ordering::Relaxed), eng.shared().cpu_misses.load(std::sync::atomic::Ordering::Relaxed));
        loop {
            // take every request that has come in; block when there is nothing to do
            if running.is_empty() && waiting.is_empty() {
                match rx.recv() {
                    Ok(j) => waiting.push_back(j),
                    Err(_) => return,
                }
            }
            while let Ok(j) = rx.try_recv() {
                waiting.push_back(j);
            }
            // admit (the engine waits for the queued step before it touches a slot)
            let ta = std::time::Instant::now();
            while !free.is_empty() {
                let Some(job) = waiting.pop_front() else { break };
                if job.prompt.len() + 1 >= max_ctx {
                    let _ = job.events.send(Event::Error(format!("prompt of {} tokens does not fit the {max_ctx}-token context", job.prompt.len())));
                    continue;
                }
                // KV pages: the running sequences' growth to their limits is reserved; the newcomer needs its
                // whole span (the pool may be smaller than slots x contexts, see QX_KV_GIB)
                let (kv_free, kv_evictable, kv_total) = eng.kv_pages();
                let reserved: usize = running.iter().map(|r| eng.kv_pages_for(r.job.prompt.len() + r.job.max_tokens).saturating_sub(eng.kv_slot_pages(r.slot))).sum();
                let need = eng.kv_pages_for(job.prompt.len() + job.max_tokens);
                if kv_free + kv_evictable < reserved + need {
                    if running.is_empty() || need > kv_total {
                        let _ = job.events.send(Event::Error(format!("the request needs {need} KV pages ({} tokens); {} are available", job.prompt.len() + job.max_tokens, kv_free + kv_evictable)));
                        continue;
                    }
                    waiting.push_front(job);
                    break;
                }
                let slot = free.pop().unwrap();
                let t0 = std::time::Instant::now();
                // a speculative request is joined: it continues batched
                for r in running.iter_mut().filter(|r| r.spec.is_some()) {
                    eng.select(r.slot);
                    eng.spec_to_batch();
                    r.spec = None;
                }
                // a constrained request compiles its grammar first (and never decodes speculatively)
                let matcher = match &job.grammar {
                    Some(spec) => match grammars.matcher(spec) {
                        Ok(m) => Some(m),
                        Err(e) => {
                            let _ = job.events.send(Event::Error(format!("{e:#}")));
                            free.push(slot);
                            continue;
                        }
                    },
                    None => None,
                };
                let alone = running.is_empty() && waiting.is_empty() && matcher.is_none();
                // a cached prefix of the prompt restarts the sequence from its end
                let hit = eng.prefix_lookup(&job.prompt, PREFIX_MIN);
                if let Some((e, _)) = hit {
                    eng.prefix_restore(slot, e, &job.prompt, &job.sampling);
                }
                let spec = if alone && eng.spec_enabled() {
                    if hit.is_none() {
                        eng.select(slot);
                        eng.reset();
                    }
                    let k = eng.drafter_k();
                    Some(eng.spec_begin(&job.prompt, &job.sampling, k))
                } else {
                    if hit.is_none() {
                        // the prompt is read in chunks between steps, by the loop below
                        eng.begin_seq_deferred(slot, &job.prompt, &job.sampling, true);
                    }
                    None
                };
                running.push(Running { job, slot, detok: Detok { tok, ids: Vec::new(), sent: 0 }, text: String::new(), n: 0, t0, t_prompt: None, finish: None, spec, prompt_saved: false, matcher, lane: None });
            }
            st_admit += ta.elapsed().as_secs_f64();
            if running.is_empty() {
                continue;
            }
            let ts = std::time::Instant::now();
            if running.len() == 1 && running[0].spec.is_some() && running[0].matcher.is_none() {
                // one request: speculative windows, synchronous
                let r = &mut running[0];
                eng.select(r.slot);
                let want = r.job.max_tokens - r.n;
                let kept = eng.spec_next(r.spec.as_mut().unwrap(), want, None);
                st_spec += 1;
                let pos = eng.slot_pos(r.slot);
                for id in kept {
                    if !r.emit(id, &stop_ids, max_ctx, pos) {
                        break;
                    }
                }
            } else {
                // queue the next step: decoding requests first (one token each), then prompt chunks in what is
                // left of the window
                // a long prompt about to run its last token: save the sequence for a repeat of the prompt
                if prefix_on {
                    for r in running.iter_mut().filter(|r| r.finish.is_none() && !r.prompt_saved) {
                        if r.job.prompt.len() >= PREFIX_PROMPT_MIN && eng.slot_pos(r.slot) + 1 == r.job.prompt.len() {
                            eng.prefix_snapshot(r.slot, PREFIX_MIN);
                        }
                        if eng.slot_pos(r.slot) + 1 >= r.job.prompt.len() {
                            r.prompt_saved = true;
                        }
                    }
                }
                let cap = eng.max_window();
                let mut segs: Vec<(usize, usize)> = Vec::new();
                let mut used = 0;
                // prompt remainders first (the long ones wait for a prefill chunk below), PROMPT_LANES at most
                for r in running.iter().filter(|r| r.finish.is_none()) {
                    let p = eng.slot_pos(r.slot);
                    let left = (r.job.prompt.len() - 1).saturating_sub(p);
                    if left > 0 && (left < PREFILL_MIN_BATCH || pf_max == 0) && used < PROMPT_LANES {
                        let n = left.min(PROMPT_CHUNK).min(PROMPT_LANES - used);
                        segs.push((r.slot, n));
                        used += n;
                    }
                }
                // decoders in the lanes left, starting after the one that went last when they do not all fit
                let dec: Vec<usize> = running.iter().filter(|r| r.finish.is_none() && eng.slot_pos(r.slot) + 1 >= r.job.prompt.len()).map(|r| r.slot).collect();
                if !dec.is_empty() {
                    let start = dec.iter().position(|&s| s > rr_last).unwrap_or(0);
                    let mut placed = 0;
                    while placed < dec.len() && used < cap {
                        let slot = dec[(start + placed) % dec.len()];
                        segs.push((slot, 1));
                        used += 1;
                        placed += 1;
                        rr_last = slot;
                    }
                    if placed == dec.len() {
                        rr_last = usize::MAX;
                    }
                }
                // lanes still free go back to the prompts, in order of arrival, as many as each has left
                for r in running.iter().filter(|r| r.finish.is_none()) {
                    if used >= cap {
                        break;
                    }
                    let p = eng.slot_pos(r.slot);
                    let already: usize = segs.iter().filter(|&&(s, _)| s == r.slot).map(|&(_, n)| n).sum();
                    let left = (r.job.prompt.len() - 1).saturating_sub(p + already);
                    if left > 0 && (left + already < PREFILL_MIN_BATCH || pf_max == 0) {
                        let n = left.min(cap - used);
                        // one segment per slot: extend it, or start one
                        match segs.iter_mut().find(|(s, _)| *s == r.slot) {
                            Some(seg) => seg.1 += n,
                            None => segs.push((r.slot, n)),
                        }
                        used += n;
                    }
                }
                // token masks: a constrained request's generated token is the sample of its segment's last lane
                let constrained = running.iter().any(|r| r.matcher.is_some());
                if constrained {
                    for lane in 0..eng.max_window() {
                        eng.set_mask(lane, None);
                    }
                    let mut lane = 0;
                    for &(slot, n) in &segs {
                        lane += n;
                        let r = running.iter_mut().find(|r| r.slot == slot).unwrap();
                        r.lane = None;
                        if let Some(m) = r.matcher.as_mut() {
                            if eng.slot_pos(slot) + n >= r.job.prompt.len() {
                                match m.compute_mask() {
                                    Ok(mask) => {
                                        eng.set_mask(lane - 1, Some(mask.as_slice()));
                                        r.lane = Some(lane - 1);
                                    }
                                    Err(e) => {
                                        let _ = r.job.events.send(Event::Error(format!("grammar: {e}")));
                                        r.finish = Some("error");
                                    }
                                }
                            }
                        }
                    }
                }
                let mut next_pending = Vec::new();
                if !segs.is_empty() {
                    let te = std::time::Instant::now();
                    eng.step_segments_async(&segs);
                    let de = te.elapsed().as_secs_f64();
                    let h = hist.entry(used).or_default();
                    h.0 += 1;
                    h.1 += de;
                    st_tok += used;
                    next_pending = segs.iter().map(|&(slot, _)| (slot, eng.slot_pos(slot) - 1)).collect();
                }
                // one prefill chunk for the oldest request with a long prompt remainder, sized to the time budget
                if pf_max > 0 {
                    let pick = running.iter().filter(|r| r.finish.is_none()).map(|r| (r.slot, r.job.prompt.len() - 1 - eng.slot_pos(r.slot).min(r.job.prompt.len() - 1))).find(|&(_, left)| left >= PREFILL_MIN_BATCH);
                    if let Some((slot, left)) = pick {
                        // whole chunks (`--prefill-budget-ms` caps them once the measured rate allows smaller ones);
                        // with no other request decoding there is nobody to keep waiting, and each chunk streams the
                        // experts the cache lacks whatever its length, so the longest chunk
                        let max = eng.prefill_chunk_for(left);
                        let decoding = running.iter().any(|r| r.slot != slot && r.finish.is_none() && eng.slot_pos(r.slot) + 1 >= r.job.prompt.len());
                        let want = if decoding { ((opt.prefill_budget_ms * 1e-3 * pf_rate) as usize).clamp(PREFILL_MIN_BATCH, max) } else { max };
                        let n = left.min(want);
                        let tp = std::time::Instant::now();
                        eng.prefill_more(slot, n);
                        let dt = tp.elapsed().as_secs_f64();
                        pf_rate = 0.7 * pf_rate + 0.3 * (n as f64 / dt.max(1e-6));
                        pf_chunks += 1;
                        pf_tokens += n;
                        pf_time += dt;
                    } else {
                        // no prompt left to read: the cache gets its lent tail back
                        eng.prefill_release();
                    }
                }
                // the previous step's samples, while this one runs (QX_ASYNC=0, or a constrained request whose
                // next mask needs this step's token: this step's, waiting for it)
                if !r#async || constrained {
                    pending = std::mem::take(&mut next_pending);
                }
                let tw = std::time::Instant::now();
                for (slot, last) in pending.drain(..) {
                    // a sleep, not a spin: the pool's threads share the cores
                    while !eng.sample_ready(slot, last) {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                    let r = running.iter_mut().find(|r| r.slot == slot).unwrap();
                    // the step ran the prompt's last token or a generated one: its sample is the next token
                    if last + 1 >= r.job.prompt.len() && r.finish.is_none() {
                        let id = eng.sample_at(slot, last);
                        if let Some(m) = r.matcher.as_mut() {
                            if r.lane.is_some() && !stop_ids.contains(&id) {
                                if let Err(e) = m.consume_token(id) {
                                    let _ = r.job.events.send(Event::Error(format!("grammar: {e}")));
                                    r.finish = Some("error");
                                    continue;
                                }
                            }
                        }
                        let more = r.emit(id, &stop_ids, max_ctx, last + 1);
                        if more && r.matcher.as_ref().is_some_and(|m| m.is_stopped()) {
                            r.finish = Some("stop");
                        }
                    }
                }
                st_wait += tw.elapsed().as_secs_f64();
                pending = next_pending;
            }
            st_t += ts.elapsed().as_secs_f64();
            st_n += 1;
            if stats && st_n % 200 == 0 {
                eprintln!("sched: {st_n} steps ({st_spec} speculative), {:.1} window tokens/step, {:.2} ms/step ({:.2} waiting for samples), admitting {:.2}s", st_tok as f64 / (st_n - st_spec).max(1) as f64, 1e3 * st_t / st_n as f64, 1e3 * st_wait / st_n as f64, st_admit);
                eprintln!("  by width: {}", hist.iter().map(|(w, (n, t))| format!("{w}:{n}x{:.1}ms", 1e3 * t / *n as f64)).collect::<Vec<_>>().join(" "));
                eprintln!("  prefill: {pf_chunks} chunks, {pf_tokens} tokens, {:.1} ms each, {:.0} tok/s", 1e3 * pf_time / pf_chunks.max(1) as f64, pf_tokens as f64 / pf_time.max(1e-9));
                let (pe, ph, pt, ps, pv) = eng.prefix_stats();
                eprintln!("  prefix cache: {pe} entries, {ph} hits, {pt} tokens restored, {ps} snapshots, {pv} evictions");
                let now = (eng.cache_stats(), eng.shared().cpu_ns.load(std::sync::atomic::Ordering::Relaxed), eng.shared().cpu_misses.load(std::sync::atomic::Ordering::Relaxed));
                let ((h0, r0, w0), n0, m0) = c_last;
                let ((h1, r1, w1), n1, m1) = now;
                eprintln!("  last 200: {:.1}% hits, {:.1} misses/step, {:.1} swaps/step, {:.2} ms/step in the CPU pool", 100.0 * (h1 - h0) as f64 / (r1 - r0).max(1) as f64, (m1 - m0) as f64 / 200.0, (w1 - w0) as f64 / 200.0, (n1 - n0) as f64 * 1e-6 / 200.0);
                c_last = now;
            }
            // finished requests: their slot is free once nothing of theirs is still queued; the conversation so
            // far goes to the prefix cache for its next turn
            let mut i = 0;
            while i < running.len() {
                if running[i].finish.is_some() && !pending.iter().any(|&(s, _)| s == running[i].slot) {
                    let r = running.swap_remove(i);
                    r.done();
                    if prefix_on {
                        eng.prefix_snapshot(r.slot, PREFIX_MIN);
                    }
                    free.push(r.slot);
                } else {
                    i += 1;
                }
            }
        }
    });
    ready_rx.recv()?.map_err(|e| anyhow::anyhow!(e))?;
    Ok(tx)
}
