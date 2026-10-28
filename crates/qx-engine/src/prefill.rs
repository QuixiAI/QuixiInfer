//! Prefill: the forward pass for a chunk of consecutive forced tokens at once, on tensor-core GEMMs. It leaves
//! every piece of sequence state (DeltaNet and conv states, PLE history, FP8 KV, indexer keys, position)
//! exactly where the decode step continues from. Routed experts the VRAM cache lacks are read in place from
//! device-mapped host memory, each once per chunk and layer. Chunks produce no logits: the decode step for
//! the prompt's last token does.
use super::*;
use qx_gpu::kernel::modules::{GEMM, PREFILL};

static COPY_AT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static MAIN_AT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Expert rows per grouped-GEMM tile (GROUP_TILE in kernels/gemm.cu).
const GROUP_TILE: usize = 64;

pub(super) struct PrefillBufs {
    pub c: usize,
    pub bytes: usize,
    /// The device memory the buffers are carved from (the expert cache's tail, lent while a prompt is read).
    pub region: Option<(u64, usize)>,
    p0: DeviceBuf,
    e: DeviceBuf,
    r: DeviceBuf,
    xn: DeviceBuf,
    lo: DeviceBuf,
    gate: DeviceBuf,
    x: DeviceBuf,
    inject: DeviceBuf,
    y: DeviceBuf,
    qkv: DeviceBuf,
    z: DeviceBuf,
    beta: DeviceBuf,
    alpha: DeviceBuf,
    cv: DeviceBuf,
    o: DeviceBuf,
    on: DeviceBuf,
    qg: DeviceBuf,
    q: DeviceBuf,
    ag: DeviceBuf,
    kraw: DeviceBuf,
    knew: DeviceBuf,
    vnew: DeviceBuf,
    kr: DeviceBuf,
    iq: DeviceBuf,
    qi: DeviceBuf,
    cells: DeviceBuf,
    counts: DeviceBuf,
    scores: DeviceBuf,
    att: DeviceBuf,
    router: DeviceBuf,
    ids: DeviceBuf,
    wts: DeviceBuf,
    sorted: DeviceBuf,
    ends: DeviceBuf,
    /// The chunk's VRAM mirror of the current layer's KV (every cell before the chunk's end; see qxp_kv_mirror),
    /// its table entry and ring size in cells.
    mir_k: DeviceBuf,
    mir_v: DeviceBuf,
    mir_ks: DeviceBuf,
    mir_vs: DeviceBuf,
    mir_tab: DeviceBuf,
    mir_cells: usize,
    /// The two-pass GDN's per-chunk products for one slice of GDN_SLICE tokens: U, W (slice x vh x d), P
    /// (slice / 32 x vh x 32 x 32), g (slice x vh).
    gdn_u: DeviceBuf,
    gdn_w: DeviceBuf,
    gdn_p: DeviceBuf,
    gdn_g: DeviceBuf,
    tile_e: DeviceBuf,
    tile_r: DeviceBuf,
    tiles: DeviceBuf,
    ptrs: DeviceBuf,
    g: DeviceBuf,
    u: DeviceBuf,
    h: DeviceBuf,
    v: DeviceBuf,
    sh_g: DeviceBuf,
    sh_u: DeviceBuf,
    sh_h: DeviceBuf,
    sh_y: DeviceBuf,
    sg: DeviceBuf,
    emb: DeviceBuf,
    pkey: DeviceBuf,
    pval: DeviceBuf,
    kn: DeviceBuf,
    qn: DeviceBuf,
    pg: DeviceBuf,
    gss: DeviceBuf,
    /// Expert staging: two halves the copy stream fills in turn, the per-layer weight pointers the chunk's
    /// MoE uses (VRAM slot or staging address), and the ready / freed counters ([0], [1]).
    stage: DeviceBuf,
    stage_half: usize,
    eptrs: DeviceBuf,
    /// Routing counts per (layer, expert) over the current prefill, handed to the cache's swapper after it.
    hist: DeviceBuf,
    flags: DeviceBuf,
    /// Groups consumed, as the chunk's MoE publishes it (u64 at 0).
    pub(crate) freed: qx_gpu::mem::HostMapped,
    copy: cudarc::driver::sys::CUstream,
    /// Staging groups issued so far, over every chunk (the counters' values).
    pub(crate) generation: u32,
}

/// One staging group: experts [lo, hi) of a layer, whose missing experts fit one staging half.
struct Group {
    lo: usize,
    hi: usize,
    /// Copies: (host source, staging destination, bytes).
    copies: Vec<(u64, u64, usize)>,
}

/// Tokens per pass of the two-pass GDN (its scratch is sized for this many, whatever the chunk).
const GDN_SLICE: usize = 2048;

/// Cells one query attends to at most: the top blocks plus a partial tail block.
fn sel_cells(c: &Config) -> usize {
    c.idx_top_k + c.kpool - 1
}

impl PrefillBufs {
    /// `region`: device memory (address, bytes) the buffers are carved from (the expert cache's tail, lent while
    /// a prompt is read); None allocates them. The few that outlive a prompt are always their own allocations.
    pub fn new(cfg: &Config, c: usize, max_ctx: usize, hc_lo: usize, stage_bytes: usize, region: Option<(u64, usize)>) -> Self {
        let mut bytes = 0usize;
        let mut f = |n: usize| {
            let at = bytes;
            bytes += (n * 4).next_multiple_of(256);
            match region {
                Some((base, cap)) => {
                    assert!(bytes <= cap, "prefill buffers past their region");
                    DeviceBuf::view(base + at as u64, n * 4)
                }
                None => DeviceBuf::new(n * 4),
            }
        };
        let own = |n: usize| DeviceBuf::new(n * 4);
        let (k, ff) = (cfg.n_expert_used, cfg.ff_exp);
        let ch = 2 * cfg.ssm_k_heads * cfg.ssm_dim + cfg.ssm_v_heads * cfg.ssm_dim;
        let vd = cfg.ssm_v_heads * cfg.ssm_dim;
        let ad = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_head_kv * cfg.head_dim;
        let idd = cfg.idx_heads * cfg.idx_dim;
        let sel = sel_cells(cfg);
        let a = c * k;
        let max_tiles = a / GROUP_TILE + cfg.n_expert;
        Self {
            c,
            p0: f(1),
            e: f(c * E),
            r: f(c * HCD),
            xn: f(c * HCD),
            lo: f(c * hc_lo),
            gate: f(c * HCD),
            x: f(c * E),
            inject: f(c * HC),
            y: f(c * E),
            qkv: f(c * ch),
            z: f(c * vd),
            beta: f(c * cfg.ssm_v_heads),
            alpha: f(c * cfg.ssm_v_heads),
            cv: f(c * ch),
            o: f(c * vd),
            on: f(c * vd),
            qg: f(c * 2 * ad),
            q: f(c * ad),
            ag: f(c * ad),
            kraw: f(c * kvd),
            knew: f(c * kvd),
            vnew: f(c * kvd),
            kr: f(c * cfg.idx_dim),
            iq: f(c * idd),
            qi: f(c * idd),
            cells: f(c * sel),
            counts: f(c),
            scores: f(c * max_ctx.div_ceil(cfg.kpool)),
            att: f(c * ad),
            router: f(c * cfg.n_expert),
            ids: f(a),
            wts: f(a),
            sorted: f(a),
            ends: f(cfg.n_expert),
            gdn_u: f(c.min(GDN_SLICE) * vd),
            gdn_w: f(c.min(GDN_SLICE) * vd),
            gdn_p: f(c.min(GDN_SLICE).div_ceil(32) * cfg.ssm_v_heads * 32 * 32),
            gdn_g: f(c.min(GDN_SLICE) * cfg.ssm_v_heads),
            tile_e: f(max_tiles),
            tile_r: f(max_tiles),
            tiles: f(1),
            ptrs: f(2 * cfg.n_expert),
            g: f(a * ff),
            u: f(a * ff),
            h: f(a * ff),
            v: f(a * E),
            sh_g: f(c * ff),
            sh_u: f(c * ff),
            sh_h: f(c * ff),
            sh_y: f(c * E),
            sg: f(c),
            emb: f(c * E),
            pkey: f(c * HCD),
            pval: f(c * E),
            kn: f(c * HCD),
            qn: f(c * HCD),
            pg: f(c * HC),
            gss: f(c * HC),
            mir_k: f(((max_ctx + crate::mailbox::WMAX + 1) * kvd).div_ceil(4)),
            mir_v: f(((max_ctx + crate::mailbox::WMAX + 1) * kvd).div_ceil(4)),
            mir_ks: f((max_ctx + crate::mailbox::WMAX + 1) * cfg.n_head_kv * crate::kvpages::SCALE_GROUPS),
            mir_vs: f((max_ctx + crate::mailbox::WMAX + 1) * cfg.n_head_kv * crate::kvpages::SCALE_GROUPS),
            mir_tab: DeviceBuf::new(64),
            mir_cells: max_ctx + crate::mailbox::WMAX + 1,
            stage: f(stage_bytes / 4),
            stage_half: stage_bytes / 2 / 4096 * 4096,
            eptrs: own(2 * cfg.n_layer * cfg.n_expert),
            hist: own(cfg.n_layer * cfg.n_expert),
            flags: own(64),
            freed: qx_gpu::mem::HostMapped::new(64),
            copy: {
                qx_gpu::mem::bind();
                unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.expect("prefill copy stream")
            },
            generation: 0,
            bytes,
            region,
        }
    }
}

fn gdn_mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("QX_GDN").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// The double-buffered ldmatrix GEMM entries (`QX_GEMM2=0`: the single-stage ones; the same bits either way).
fn gemm2() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("QX_GEMM2").is_ok_and(|v| v == "0"))
}

fn grid1(n: usize, t: u32) -> (u32, u32, u32) {
    ((n as u32).div_ceil(t), 1, 1)
}

impl<'a> Engine<'a> {
    fn pk(&self, entry: &'static str, threads: u32, grid: (u32, u32, u32), info: &[u64]) {
        self.gpu.run(PREFILL.kernel(entry, CubeDim::new_1d(threads)), grid, info);
    }

    /// y[t] = m . x[t] for t < T; F32 weights stay f32, the rest run on bf16 tensor cores.
    fn gemm(&self, m: &Mat, x: u64, y: u64, t: usize) {
        if m.ty == GgmlType::F32 && m.rows < 128 && m.cols % 4 == 0 {
            self.gpu.run(GEMM.kernel("qx_gemm_f32_narrow", CubeDim::new_1d(256)), ((t as u32).div_ceil(8), m.rows as u32, 1), &[x, m.buf.ptr, y, t as u64, m.rows as u64, m.cols as u64, m.rows as u64]);
        } else if m.ty == GgmlType::F32 {
            self.gpu.run(GEMM.kernel("qx_gemm_f32", CubeDim::new_1d(256)), ((m.rows as u32).div_ceil(64), (t as u32).div_ceil(64), 1), &[x, m.buf.ptr, y, t as u64, m.rows as u64, m.cols as u64, m.rows as u64]);
        } else {
            self.gpu.run(GEMM.kernel(if gemm2() { "qx_gemm2" } else { "qx_gemm" }, CubeDim::new_1d(256)), ((m.rows as u32).div_ceil(64), (t as u32).div_ceil(128), 1), &[x, m.buf.ptr, y, t as u64, m.rows as u64, m.cols as u64, m.qx, m.stride as u64, m.rows as u64, 0]);
        }
    }

    /// Whether `prefill` can run: buffers were allocated and the KV cache is FP8.
    pub fn can_prefill(&self) -> bool {
        self.prefill.is_some() && self.state.iter().all(|s| !matches!(s, LayerState::Attn { fp8: None, .. }))
    }

    /// Appends forced tokens to the sequence and runs them with `prefill`; logits come from the next `forward`.
    pub fn prefill_forced(&mut self, toks: &[u32]) {
        self.tokens.extend_from_slice(toks);
        let all = self.tokens.clone();
        self.seq().allow_sampled.store(false, Ordering::Release);
        self.set_forced(&all);
        self.prefill(toks.len());
    }

    /// Runs the forced tokens at positions pos .. pos + n through every layer, in chunks. The tokens must
    /// already be forced (`set_forced`); afterwards the next decode step continues at pos + n.
    pub fn prefill(&mut self, n: usize) {
        assert!(self.can_prefill());
        assert!(self.pos + n <= self.max_ctx);
        // the big buffers for a long run (and while they are lent anyway), the small ones otherwise
        let small_c = self.prefill.as_ref().unwrap().c.min(self.prefill_alt.as_ref().map_or(usize::MAX, |p| p.c));
        self.prefill_tier(n > small_c || (self.prefill_big && self.lent.get()));
        self.prefill_lend();
        let c = self.prefill.as_ref().unwrap().c;
        self.prefill.as_ref().unwrap().hist.zero();
        let mut done = 0;
        while done < n {
            let t = (n - done).min(c);
            self.prefill_chunk(t);
            done += t;
            // a prefix cache entry at every chunk boundary but the last (the caller takes that one)
            if self.snapshot_prefill && done < n {
                self.prefix_snapshot(self.cur, 1);
            }
        }
        if !self.keep_lent {
            self.prefill_release();
        }
        // The prompt's routing, for the adaptive cache to act on before decode settles it token by token.
        let pb = self.prefill.as_ref().unwrap();
        let mut counts: Vec<u32> = pb.hist.download(self.cfg.n_layer * self.cfg.n_expert);
        // weighed as at most one small-tier chunk: the swapper adds these to its scores, and a big chunk's
        // counts would outweigh the decode history (the cache then serves the prompt's experts, not the answer's)
        if n > small_c {
            for c in counts.iter_mut().filter(|c| **c > 0) {
                *c = ((*c as u64 * small_c as u64 / n as u64) as u32).max(1);
            }
        }
        if self.prefill_feeds_cache {
            *self.driver.shared.bulk.lock().unwrap() = Some(counts);
        }
    }

    fn prefill_chunk(&mut self, t: usize) {
        let p0 = self.pos;
        let dbg = std::env::var("QX_PREFILL_DEBUG").is_ok();
        if dbg { eprintln!("chunk {p0}+{t}"); }
        self.gpu.sync();
        self.touch_pages(self.cur, p0 + t);
        let plan = self.stage_plan();
        let g0 = self.prefill.as_ref().unwrap().generation;
        self.prefill.as_mut().unwrap().generation = g0.wrapping_add(plan.iter().map(|p| p.len() as u32).sum());
        {
            let pb = self.prefill.as_ref().unwrap();
            pb.p0.upload(&[p0 as u32]);
            let copy = pb.copy as usize;
            let ready = &self.stage_ready;
            let freed = pb.freed.atomic_u64(0);
            let plan_ref = &plan;
            let done = std::sync::atomic::AtomicBool::new(false);
            let done_ref = &done;
            std::thread::scope(|sc| {
            sc.spawn(move || Self::copy_groups(plan_ref, g0, copy, ready, freed));
            if std::env::var("QX_PREFILL_WATCH").is_ok() {
                sc.spawn(move || {
                    let mut n = 0;
                    while !done_ref.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        n += 1;
                        if n % 30 == 0 {
                            eprintln!("watch: copy at {} main at {} ready {} freed {} (g0 {g0})", COPY_AT.load(Ordering::Relaxed), MAIN_AT.load(Ordering::Relaxed), ready.load(Ordering::Relaxed), freed.load(Ordering::Relaxed));
                        }
                    }
                });
            }
            let mut gi = g0;
            let tt = t as u64;
            // the chunk's token embeddings, gathered by the table's own type (Q6_K in the IQ3_XXS GGUF, Q8_0 in the NVFP4 one)
            ops::dequant_gather(&self.gpu, self.embd.ty, self.embd.buf.ptr, pb.e.ptr, t, E, self.ctx.ptr + (self.ctx_at(self.cur) + p0 * 4) as u64);
            self.pk("qxp_hc_init", 256, grid1(t * HCD, 256), &[pb.r.ptr, pb.e.ptr, E as u64, HC as u64, tt]);
            let nl = self.layers.len();
            for l in 0..nl {
                let w = &self.layers[l];
                if let Some(p) = &w.ple {
                    self.prefill_ple(p, t);
                }
                // The previous layer's combine normalised the residual for this mix, unless PLE changed it since.
                if l == 0 || w.ple.is_some() {
                    self.pk("qxp_rms", 1024, (HC as u32, t as u32, 1), &[pb.r.ptr, 0, w.hc_attn.norm.ptr, pb.xn.ptr, E as u64, E as u64, bits(EPS), HCD as u64, HCD as u64]);
                }
                self.prefill_mix(&w.hc_attn, t);
                match &w.mixer {
                    MixerW::Gdn(gw) => self.prefill_gdn(gw, &self.state[l], t),
                    MixerW::Attn(aw) => self.prefill_attn(aw, &self.state[l], t),
                }
                self.pk("qxp_hc_combine_norm", 1024, (HC as u32, t as u32, 1), &[pb.r.ptr, pb.y.ptr, pb.inject.ptr, w.hc_ffn.norm.ptr, pb.xn.ptr, E as u64, bits(EPS), HC as u64]);
                self.prefill_mix(&w.hc_ffn, t);
                self.prefill_moe(w, l, t, &plan[l], gi);
                gi = gi.wrapping_add(plan[l].len() as u32);
                if std::env::var("QX_PREFILL_DEBUG").is_ok() {
                    self.gpu.sync();
                    let f: Vec<u32> = pb.flags.download(2);
                    eprintln!("prefill layer {l}: {} groups, flags {:?}", plan[l].len(), f);
                }
                let next_norm = if l + 1 < nl { &self.layers[l + 1].hc_attn.norm } else { &self.head.norm };
                self.pk("qxp_hc_combine_norm", 1024, (HC as u32, t as u32, 1), &[pb.r.ptr, pb.y.ptr, pb.inject.ptr, next_norm.ptr, pb.xn.ptr, E as u64, bits(EPS), HC as u64]);
            }
            // the drafter's KV for the chunk: the tokens at p0 + 1 .. p0 + t with the chunk's residuals
            if self.drafter.is_some() && self.draft_slot == Some(self.cur) {
                // the step state is still at p0 during the chunk
                self.draft_catch_up(p0, t, pb.r.ptr, self.ctx.ptr + (self.ctx_at(self.cur) + (p0 + 1) * 4) as u64, false, p0);
            }
            MAIN_AT.store(u32::MAX, Ordering::Relaxed);
            self.gpu.sync();
            done.store(true, Ordering::Relaxed);
            });
        }
        // Hand the position on: the device step state, and the mapped step count the CPU threads watch.
        self.gpu.sync();
        let mut st = self.step_words();
        st[0] = (p0 + t) as u32;
        self.set_step_words(&st);
        self.driver.shared.count(self.cur).store((p0 + t) as u64, Ordering::Release);
        self.pos = p0 + t;
    }

    /// Splits each layer's experts into staging groups and writes the per-layer weight pointers: a VRAM cache
    /// slot, or the staging address the expert's group copies it to. A chunk of more than a few dozen tokens
    /// routes to nearly every expert of every layer, so every missing expert is staged without looking at the
    /// routing.
    fn stage_plan(&self) -> Vec<Vec<Group>> {
        let pb = self.prefill.as_ref().unwrap();
        let ne = self.cfg.n_expert;
        let table: Vec<u64> = self.cache.table.download(self.cfg.n_layer * ne);
        let mut ptrs = vec![0u64; self.cfg.n_layer * ne];
        let mut plan = Vec::new();
        let mut gi = 0usize;
        for l in 0..self.cfg.n_layer {
            let (stride, rec) = (self.host.layers[l].1, self.host.record_bytes(l));
            let cap = (pb.stage_half / stride).max(1);
            let mut groups = Vec::new();
            let mut cur = Group { lo: 0, hi: 0, copies: Vec::new() };
            let mut used = 0usize;
            for e in 0..ne {
                let slot = table[l * ne + e];
                if slot != 0 {
                    ptrs[l * ne + e] = slot;
                } else {
                    if used == cap {
                        cur.hi = e;
                        groups.push(std::mem::replace(&mut cur, Group { lo: e, hi: 0, copies: Vec::new() }));
                        gi += 1;
                        used = 0;
                    }
                    let dst = pb.stage.ptr + ((gi % 2) * pb.stage_half + used * stride) as u64;
                    let src = self.host.record(l, e) as u64;
                    match cur.copies.last_mut() {
                        Some(c) if c.0 + c.2 as u64 + (stride - rec) as u64 == src && c.1 + c.2 as u64 + (stride - rec) as u64 == dst => c.2 += stride,
                        _ => cur.copies.push((src, dst, rec)),
                    }
                    ptrs[l * ne + e] = dst;
                    used += 1;
                }
            }
            cur.hi = ne;
            groups.push(cur);
            gi += 1;
            plan.push(groups);
        }
        pb.eptrs.upload(&ptrs);
        plan
    }

    /// The copy thread's work for one chunk: group g waits until group g - 2 (same staging half) is consumed,
    /// is copied on the copy stream, then publishes ready = g + 1. The MoE's launches wait for it on the host:
    /// a GPU spin-wait would deadlock against a first-launch module load, which holds a driver lock until
    /// the GPU idles while the copy thread needs that lock.
    fn copy_groups(plan: &[Vec<Group>], g0: u32, copy: usize, ready: &std::sync::atomic::AtomicU32, freed: &std::sync::atomic::AtomicU64) {
        use cudarc::driver::sys::*;
        qx_gpu::mem::bind();
        let copy = copy as CUstream;
        let mut g = g0;
        for grp in plan.iter().flatten() {
            if g.wrapping_sub(g0) >= 2 {
                let want = g.wrapping_sub(1) as u64;
                let mut spins = 0u32;
                while ((freed.load(Ordering::Acquire) as u32).wrapping_sub(want as u32) as i32) < 0 {
                    spins += 1;
                    if spins > 1000 {
                        std::thread::sleep(std::time::Duration::from_micros(20));
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }
            COPY_AT.store(g, Ordering::Relaxed);
            unsafe {
                for &(src, dst, n) in &grp.copies {
                    cuMemcpyHtoDAsync_v2(dst, src as *const _, n, copy).result().expect("stage copy");
                }
                cuStreamSynchronize(copy).result().expect("copy stream");
            }
            ready.store(g.wrapping_add(1), Ordering::Release);
            g = g.wrapping_add(1);
        }
    }

    fn prefill_ple(&self, p: &PleW, t: usize) {
        let pb = self.prefill.as_ref().unwrap();
        let tt = t as u64;
        let sh = &self.driver.shared;
        self.pk("qxp_ple_wait", 1024, (1, 1, 1), &[sh.ple.dptr + sh.ple_at(self.cur) as u64, pb.p0.ptr, pb.emb.ptr, E as u64, tt, mailbox::PLE_CAP as u64]);
        self.gemm(&p.key, pb.emb.ptr, pb.pkey.ptr, t);
        self.gemm(&p.value, pb.emb.ptr, pb.pval.ptr, t);
        self.pk("qxp_rms", 1024, (HC as u32, t as u32, 1), &[pb.pkey.ptr, 0, p.norm_key.ptr, pb.kn.ptr, E as u64, E as u64, bits(EPS), HCD as u64, HCD as u64]);
        self.pk("qxp_rms", 1024, (HC as u32, t as u32, 1), &[pb.r.ptr, 0, p.norm_query.ptr, pb.qn.ptr, E as u64, E as u64, bits(EPS), HCD as u64, HCD as u64]);
        self.pk("qxp_ple_gate", 256, (HC as u32, t as u32, 1), &[pb.kn.ptr, pb.qn.ptr, pb.pg.ptr, E as u64]);
        self.pk("qxp_ple_gss", 256, (HC as u32, t as u32, 1), &[pb.pval.ptr, pb.pg.ptr, pb.gss.ptr, E as u64]);
        self.pk("qxp_ple_apply", 256, grid1(HCD, 256), &[pb.pval.ptr, pb.pg.ptr, pb.gss.ptr, p.norm_conv.ptr, p.conv.ptr, self.s.hist.ptr + (self.cur * self.strides.hist * 4) as u64, pb.r.ptr, E as u64, bits(EPS), tt, HC as u64]);
    }

    /// HC mix of the chunk from `xn` into `x` (and `inject`).
    fn prefill_mix(&self, h: &HcW, t: usize) {
        let pb = self.prefill.as_ref().unwrap();
        self.gemm(&h.down, pb.xn.ptr, pb.lo.ptr, t);
        if let Some(inj) = &h.inject {
            self.gemm(inj, pb.xn.ptr, pb.inject.ptr, t);
        }
        self.pk("qxp_scale_silu", 256, grid1(t * h.down.rows, 256), &[pb.lo.ptr, (t * h.down.rows) as u64, bits(1.0 / HC as f32)]);
        self.gemm(&h.up, pb.lo.ptr, pb.gate.ptr, t);
        self.pk("qxp_hc_mix_out", 256, grid1(t * E, 256), &[pb.xn.ptr, pb.gate.ptr, pb.x.ptr, E as u64, HC as u64, t as u64]);
    }

    fn prefill_gdn(&self, gw: &GdnW, st: &LayerState, t: usize) {
        let (c, pb) = (&self.cfg, self.prefill.as_ref().unwrap());
        let LayerState::Gdn { conv, s: sst, .. } = st else { unreachable!() };
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_dim);
        let ch = gw.qkv.rows;
        let tt = t as u64;
        // the current slot's states
        let conv = conv.ptr + (self.cur * self.strides.conv * 4) as u64;
        let sst = sst.ptr + (self.cur * self.strides.gdn * 4) as u64;
        self.gemm(&gw.qkv, pb.x.ptr, pb.qkv.ptr, t);
        self.gemm(&gw.gate, pb.x.ptr, pb.z.ptr, t);
        self.gemm(&gw.beta, pb.x.ptr, pb.beta.ptr, t);
        self.gemm(&gw.alpha, pb.x.ptr, pb.alpha.ptr, t);
        self.pk("qxp_gdn_conv", d as u32, ((ch / d) as u32, t as u32, 1), &[pb.qkv.ptr, 0, gw.conv.ptr, pb.cv.ptr, ch as u64, d as u64, kh as u64, bits(EPS), conv, tt]);
        self.pk("qxp_gdn_conv_state", 256, grid1(ch, 256), &[pb.qkv.ptr, conv, ch as u64, tt]);
        // QX_GDN: 0 token by token, 1 chunked in one kernel, 2 (default) chunked in two passes over slices
        match gdn_mode() {
            0 | 1 => {
                let k = if gdn_mode() == 0 { "qxp_gdn_recur" } else { "qxp_gdn_chunk" };
                self.pk(k, if gdn_mode() == 0 { 128 } else { 256 }, (vh as u32, 4, 1), &[pb.cv.ptr, sst, pb.o.ptr, pb.beta.ptr, pb.alpha.ptr, gw.dt_bias.ptr, gw.a.ptr, kh as u64, d as u64, tt, ch as u64, vh as u64]);
            }
            _ => {
                for s0 in (0..t).step_by(GDN_SLICE) {
                    let n = (t - s0).min(GDN_SLICE);
                    let (cv, bt, al, o) = (pb.cv.ptr + (s0 * ch * 4) as u64, pb.beta.ptr + (s0 * vh * 4) as u64, pb.alpha.ptr + (s0 * vh * 4) as u64, pb.o.ptr + (s0 * vh * d * 4) as u64);
                    self.pk("qxp_gdn_prep", 256, (n.div_ceil(32) as u32, vh as u32, 1), &[cv, pb.gdn_u.ptr, pb.gdn_w.ptr, pb.gdn_p.ptr, pb.gdn_g.ptr, bt, al, gw.dt_bias.ptr, gw.a.ptr, kh as u64, d as u64, n as u64, ch as u64, vh as u64]);
                    self.pk("qxp_gdn_scan", 256, (vh as u32, 4, 1), &[cv, sst, o, pb.gdn_u.ptr, pb.gdn_w.ptr, pb.gdn_p.ptr, pb.gdn_g.ptr, kh as u64, d as u64, n as u64, ch as u64, vh as u64]);
                }
            }
        }
        self.pk("qxp_gdn_out", 128, (vh as u32, t as u32, 1), &[pb.o.ptr, gw.norm.ptr, pb.z.ptr, pb.on.ptr, d as u64, bits(EPS), vh as u64]);
        self.gemm(&gw.out, pb.on.ptr, pb.y.ptr, t);
    }

    fn prefill_attn(&self, aw: &AttnW, st: &LayerState, t: usize) {
        let (c, pb) = (&self.cfg, self.prefill.as_ref().unwrap());
        let LayerState::Attn { kr, kp, fp8, .. } = st else { unreachable!() };
        let f = fp8.as_ref().expect("prefill needs the FP8 KV cache");
        // the current slot's keys and KV
        let kr = kr.ptr + (self.cur * self.strides.kr * 4) as u64;
        let kp = kp.ptr + (self.cur * self.strides.kp * 4) as u64;
        let kvt = f.tab.ptr + (self.cur * 64) as u64;
        let (nh, nkv, hd, id) = (c.n_head, c.n_head_kv, c.head_dim, c.idx_dim);
        let tt = t as u64;
        self.gemm(&aw.q, pb.x.ptr, pb.qg.ptr, t);
        self.gemm(&aw.k, pb.x.ptr, pb.kraw.ptr, t);
        self.gemm(&aw.v, pb.x.ptr, pb.vnew.ptr, t);
        self.gemm(&aw.ik, pb.x.ptr, pb.kr.ptr, t);
        self.gemm(&aw.iq, pb.x.ptr, pb.iq.ptr, t);
        #[allow(clippy::too_many_arguments)]
        let nr = |x: u64, xh: usize, xrow: usize, wv: u64, out: u64, oh: usize, orow: usize, heads: usize, dim: usize| {
            self.pk("qxp_norm_rope", dim as u32, (heads as u32, t as u32, 1), &[x, pb.p0.ptr, wv, out, xh as u64, dim as u64, c.n_rot as u64, xrow as u64, bits(c.rope_base), bits(EPS), oh as u64, orow as u64]);
        };
        nr(pb.qg.ptr, 2 * hd, 2 * nh * hd, aw.q_norm.ptr, pb.q.ptr, hd, nh * hd, nh, hd);
        self.pk("qxp_take_halves", 256, grid1(t * nh * hd, 256), &[pb.qg.ptr, pb.ag.ptr, hd as u64, nh as u64, tt]);
        nr(pb.kraw.ptr, hd, nkv * hd, aw.k_norm.ptr, pb.knew.ptr, hd, nkv * hd, nkv, hd);
        self.pk("qxp_kv_append_fp8", (hd / 2) as u32, (nkv as u32, 2, t as u32), &[pb.knew.ptr, pb.vnew.ptr, kvt, 0, 0, 0, 0, 0, 0, 0, pb.p0.ptr, self.kv_window as u64, nkv as u64, hd as u64]);
        // Indexer: pooled keys of the blocks the chunk completes (before the ring moves on), then the ring.
        self.pk("qxp_idx_pool", id as u32, ((t / c.kpool + 2) as u32, 1, 1), &[pb.kr.ptr, pb.p0.ptr, aw.ik_norm.ptr, kp, id as u64, c.n_rot as u64, bits(c.rope_base), bits(EPS), c.kpool as u64, kr, tt, IDX_RING as u64]);
        self.pk("qxp_idx_ring", id as u32, (1, 1, 1), &[pb.kr.ptr, pb.p0.ptr, kr, id as u64, IDX_RING as u64, tt]);
        nr(pb.iq.ptr, id, c.idx_heads * id, aw.iq_norm.ptr, pb.qi.ptr, id, c.idx_heads * id, c.idx_heads, id);
        let sel = sel_cells(c);
        let max_blocks = self.max_ctx.div_ceil(c.kpool);
        let top = (c.idx_top_k / c.kpool) as u64;
        // the scores as one tiled product over the chunk (QX_IDX_TILED=0: each query's block scores them itself)
        let tiled = !std::env::var("QX_IDX_TILED").is_ok_and(|v| v == "0");
        if tiled {
            let nv_hi = (self.pos + t) / c.kpool;
            self.pk("qxp_idx_score", 256, ((nv_hi as u32).div_ceil(64).max(1), (t as u32).div_ceil(64), 1), &[pb.qi.ptr, pb.p0.ptr, kp, pb.scores.ptr, c.idx_heads as u64, id as u64, c.kpool as u64, top, max_blocks as u64, tt]);
        }
        self.pk("qxp_qsa_select", 1024, (t as u32, 1, 1), &[pb.qi.ptr, pb.p0.ptr, kp, pb.cells.ptr, pb.counts.ptr, c.idx_heads as u64, id as u64, c.kpool as u64, top, pb.scores.ptr, max_blocks as u64, sel as u64, tiled as u64]);
        // cells older than the VRAM ring: mirror the layer's KV into VRAM for the chunk's attention
        let end = self.pos + t;
        let (kvt, ring) = if end + crate::mailbox::WMAX + 1 > self.kv_window && !std::env::var("QX_KV_MIRROR").is_ok_and(|v| v == "0") {
            self.gpu.run(ATTN.kernel("qxp_kv_mirror", CubeDim::new_1d((hd / 2) as u32)), (end as u32, 2, 1), &[kvt, pb.mir_tab.ptr, pb.mir_k.ptr, pb.mir_v.ptr, pb.mir_ks.ptr, pb.mir_vs.ptr, end as u64, (end - 1) as u64, self.kv_window as u64, nkv as u64, hd as u64]);
            (pb.mir_tab.ptr, pb.mir_cells)
        } else {
            (kvt, self.kv_window)
        };
        self.gpu.run(
            ATTN.kernel(match nh / nkv { 8 => "qxp_attn_fp8_g8", 12 => "qxp_attn_fp8_g12", 16 => "qxp_attn_fp8_g16", g => panic!("no FP8 attention kernel for {g} heads per kv head") }, CubeDim::new_1d(256)),
            (t as u32, nkv as u32, 1),
            &[pb.q.ptr, pb.cells.ptr, pb.counts.ptr, pb.att.ptr, nkv as u64, nh as u64, hd as u64, bits(1.0 / (hd as f32).sqrt()), kvt, 0, 0, 0, 0, 0, 0, 0, pb.p0.ptr, ring as u64, sel as u64, tt - 1],
        );
        self.pk("qxp_mul_sigmoid", 256, grid1(t * nh * hd, 256), &[pb.att.ptr, pb.ag.ptr, (t * nh * hd) as u64]);
        self.gemm(&aw.o, pb.att.ptr, pb.y.ptr, t);
    }

    fn prefill_moe(&self, w: &LayerW, l: usize, t: usize, groups: &[Group], g0: u32) {
        let (c, pb) = (&self.cfg, self.prefill.as_ref().unwrap());
        let (ff, k, ne) = (c.ff_exp, c.n_expert_used, c.n_expert);
        let a = t * k;
        self.gemm(&w.router, pb.x.ptr, pb.router.ptr, t);
        self.pk("qxp_route", 32, (t as u32, 1, 1), &[pb.router.ptr, ne as u64, k as u64, pb.ids.ptr, pb.wts.ptr]);
        let le = &w.experts;
        self.pk("qxp_moe_group", 1024, (1, 1, 1), &[pb.ids.ptr, a as u64, ne as u64, pb.sorted.ptr, pb.ends.ptr, pb.tile_e.ptr, pb.tile_r.ptr, pb.tiles.ptr, pb.ptrs.ptr, pb.eptrs.ptr + (l * ne * 8) as u64, 0, 0, GROUP_TILE as u64, pb.hist.ptr + (l * ne * 4) as u64]);
        let max_tiles = (a / GROUP_TILE + ne) as u32;
        let gq = ops::qx_type(le.gate_ty());
        // NVFP4 records end in the per-tensor scales; `which` picks gate, up or down
        let tail = if le.tail_bytes() != 0 { (2 * le.gate_bytes() + le.down_bytes()) as u64 } else { 0 };
        let grouped = |x: u64, y: u64, n: usize, kk: usize, ty: u64, rb: usize, off: usize, gather: u64, scatter: u64, lo: usize, hi: usize, which: u64| {
            self.gpu.run(GEMM.kernel(if gemm2() { "qx_gemm_grouped2" } else { "qx_gemm_grouped" }, CubeDim::new_1d(128)), (max_tiles, (n as u32).div_ceil(64), 1), &[x, pb.ptrs.ptr, y, pb.sorted.ptr, pb.tile_e.ptr, pb.tile_r.ptr, pb.tiles.ptr, pb.ends.ptr, n as u64, kk as u64, ty, rb as u64, off as u64, gather, scatter, k as u64, lo as u64, hi as u64, tail, which]);
        };
        let freed = pb.freed.dptr;
        for (i, grp) in groups.iter().enumerate() {
            let g = g0.wrapping_add(i as u32);
            MAIN_AT.store(g, Ordering::Relaxed);
            if !grp.copies.is_empty() {
                // the earlier launches must reach the GPU: the copy thread waits on them
                self.gpu.client.flush().expect("flush");
                while (self.stage_ready.load(Ordering::Acquire).wrapping_sub(g.wrapping_add(1)) as i32) < 0 {
                    std::hint::spin_loop();
                }
            }
            grouped(pb.x.ptr, pb.g.ptr, ff, E, gq, le.gate.row_bytes(), 0, 1, 0, grp.lo, grp.hi, 0);
            grouped(pb.x.ptr, pb.u.ptr, ff, E, gq, le.gate.row_bytes(), le.gate_bytes(), 1, 0, grp.lo, grp.hi, 1);
            self.pk("qxp_silu_mul2_range", 256, grid1(a * ff, 256), &[pb.g.ptr, pb.u.ptr, pb.h.ptr, pb.ends.ptr, grp.lo as u64, grp.hi as u64, ff as u64, a as u64]);
            grouped(pb.h.ptr, pb.v.ptr, E, ff, ops::qx_type(le.down.ty), le.down.row_bytes(), 2 * le.gate_bytes(), 0, 1, grp.lo, grp.hi, 2);
            self.pk("qxp_signal", 1, (1, 1, 1), &[freed, g.wrapping_add(1) as u64]);
        }
        // shared expert
        self.gemm(&w.sh_gate, pb.x.ptr, pb.sh_g.ptr, t);
        self.gemm(&w.sh_up, pb.x.ptr, pb.sh_u.ptr, t);
        self.pk("qxp_silu_mul2", 256, grid1(t * ff, 256), &[pb.sh_g.ptr, pb.sh_u.ptr, pb.sh_h.ptr, (t * ff) as u64]);
        self.gemm(&w.sh_down, pb.sh_h.ptr, pb.sh_y.ptr, t);
        self.gemm(&w.sh_router, pb.x.ptr, pb.sg.ptr, t);
        self.pk("qxp_moe_combine", 256, grid1(t * E, 256), &[pb.v.ptr, pb.wts.ptr, pb.y.ptr, E as u64, k as u64, pb.sh_y.ptr, pb.sg.ptr, t as u64]);
    }
}
