//! The MTP drafter: the published one-layer next-token module (`mtp-*.gguf`), all of it in VRAM. For the token at
//! position p it takes the trunk's residual at p - 1 and the token's embedding, runs one full-attention layer and
//! one MoE layer with its own weights and KV, and proposes the token at p + 1 through the trunk's head (the
//! drafter's embedding and head are the trunk's weights, quantized differently). Its precision only affects how
//! often drafts are kept: verification decides every token.
use super::*;

pub(super) struct Drafter {
    hc_attn: HcW,
    hc_ffn: HcW,
    attn: AttnW,
    eh_norm_e: DeviceBuf,
    eh_norm_h: DeviceBuf,
    hc_head: HcW,
    experts: DeviceBuf,
    rec_stride: usize,
    gate_bytes: usize,
    gate_row: usize,
    down_row: usize,
    state: LayerState,
    groups: Groups,
    head_hc: (MGemv, MGemv),
    /// eh_proj over 8 input rows at a time (a window's tokens x streams), each writing its own rows of `r`.
    eh: Vec<MGemv>,
    /// The drafter's residual (WMAX x HCD), its input rows (WMAX x HC x 2E), its first position (u32), and the
    /// drafts with their probabilities (u32 / f32 x WMAX).
    r: DeviceBuf,
    cat: DeviceBuf,
    pub pos: DeviceBuf,
    /// Lanes for its windows: every token in slot 0 of its own (single-sequence) state.
    pub slot0: DeviceBuf,
    /// Its host KV pages and their table.
    _pages: (crate::kvpages::KvPages, DeviceBuf),
    pub out: DeviceBuf,
    pub prob: DeviceBuf,
    pub bytes: usize,
    /// Matrices only the fused launches above point at.
    _mats: Vec<Mat>,
    /// The draft head: the trunk's output rows for the first `vocab` tokens (the most frequent: ids follow
    /// merge order), and how many drafts a pass makes.
    head: Option<MGemv>,
    pub vocab: usize,
    pub k: usize,
    /// Captured draft passes by catch-up width (index t): catch-up of t tokens, then k - 1 chained steps.
    pub graphs: Vec<Option<cubecl_runtime::client::Graph>>,
}

impl Drafter {
    pub fn load(path: &std::path::Path, cfg: &Config, s: &Scratch, max_ctx: usize, window: usize) -> Result<Self> {
        let g = Gguf::open(path)?;
        let l = cfg.n_layer;
        let bytes = std::cell::Cell::new(0usize);
        let t = |n: &str| g.tensor(n).ok_or_else(|| anyhow!("drafter: missing tensor {n}"));
        let mat = |n: &str| -> Result<Mat> {
            let w = W::new(&g, t(n)?);
            let (b, qx, stride) = ops::to_planar(w.ty, w.data, w.rows, w.cols);
            bytes.set(bytes.get() + b.len());
            Ok(Mat { buf: DeviceBuf::from_slice(&b), ty: w.ty, rows: w.rows, cols: w.cols, qx, stride })
        };
        let vecf = |n: &str| -> Result<DeviceBuf> { Ok(DeviceBuf::from_slice(&W::new(&g, t(n)?).to_vec())) };
        let b = |x: &str| format!("blk.{l}.{x}");
        let hc = |p: &str, inject: bool| -> Result<HcW> {
            Ok(HcW { norm: vecf(&format!("{p}_norm.weight"))?, down: mat(&format!("{p}_down.weight"))?, up: mat(&format!("{p}_up.weight"))?, inject: if inject { Some(mat(&format!("{p}_inject.weight"))?) } else { None } })
        };
        let hc_attn = hc(&b("hc_attn"), true)?;
        let hc_ffn = hc(&b("hc_ffn"), true)?;
        let hc_head = hc(&b("nextn.hc_head"), false)?;
        let attn = AttnW {
            q: mat(&b("attn_q.weight"))?,
            k: mat(&b("attn_k.weight"))?,
            v: mat(&b("attn_v.weight"))?,
            o: mat(&b("attn_output.weight"))?,
            q_norm: vecf(&b("attn_q_norm.weight"))?,
            k_norm: vecf(&b("attn_k_norm.weight"))?,
            iq: mat(&b("indexer.q_proj.weight"))?,
            ik: mat(&b("indexer.k_proj.weight"))?,
            iq_norm: vecf(&b("indexer.q_norm.weight"))?,
            ik_norm: vecf(&b("indexer.k_norm.weight"))?,
        };
        let router = mat(&b("ffn_gate_inp.weight"))?;
        let sh_gate = mat(&b("ffn_gate_shexp.weight"))?;
        let sh_up = mat(&b("ffn_up_shexp.weight"))?;
        let sh_down = mat(&b("ffn_down_shexp.weight"))?;
        let sh_router = mat(&b("ffn_gate_inp_shexp.weight"))?;
        let eh_proj = mat(&b("nextn.eh_proj.weight"))?;
        let eh_norm_e = vecf(&b("nextn.enorm.weight"))?;
        let eh_norm_h = vecf(&b("nextn.hnorm.weight"))?;
        // every expert resident: records [gate | up | down], requantized from Q8_0 to IQ4_NL (half the VRAM;
        // the drafter's precision only changes which drafts are kept)
        let le = crate::experts::LayerExperts::load(&g, l, cfg.ff_exp, E)?;
        if le.gate_ty() != GgmlType::Q8_0 || le.down.ty != GgmlType::Q8_0 {
            return Err(anyhow!("drafter experts are {:?}/{:?}, expected Q8_0", le.gate_ty(), le.down.ty));
        }
        let ff = cfg.ff_exp;
        let (gate_row, down_row) = (E / 32 * 18, ff / 32 * 18);
        let (gate_bytes, down_bytes) = (ff * gate_row, E * down_row);
        let rec = (2 * gate_bytes + down_bytes).next_multiple_of(256);
        // cached across starts: the requantization takes several seconds
        let meta = std::fs::metadata(path)?;
        let stamp = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let cache_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/quixiinfer");
        let cache_file = cache_dir.join(format!("drafter-experts-iq4nl-{}-{stamp}.bin", meta.len()));
        let cached = std::fs::read(&cache_file).ok().filter(|b| b.len() == rec * cfg.n_expert);
        let have = cached.is_some();
        let mut host = cached.unwrap_or_else(|| vec![0u8; rec * cfg.n_expert]);
        if !have {
            use rayon::prelude::*;
            host.par_chunks_mut(rec).enumerate().for_each(|(e, dst)| {
                let parts = le.parts(e);
                let mut o = 0;
                for (pi, part) in parts.iter().enumerate() {
                    let cols = if pi < 2 { E } else { ff };
                    let src_row = cols / 32 * 34;
                    let dst_row = cols / 32 * 18;
                    let mut f = vec![0f32; cols];
                    for r in 0..part.len() / src_row {
                        qx_quant::dequant_q8_0(&part[r * src_row..(r + 1) * src_row], &mut f);
                        qx_quant::quantize_iq4_nl(&f, &mut dst[o..o + dst_row]);
                        o += dst_row;
                    }
                }
            });
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cache_file, &host);
        }
        let experts = DeviceBuf::from_slice(&host);
        bytes.set(bytes.get() + host.len());
        let (nkv, hd) = (cfg.n_head_kv, cfg.head_dim);
        // its own host pages for its one attention layer, all assigned up front
        let mut pages = crate::kvpages::KvPages::new(1, nkv, hd, 1, max_ctx, 0, None);
        let pages_tab = DeviceBuf::new(pages.max_pages * 8);
        let ids: Vec<u32> = (0..pages.max_pages).map(|_| pages.alloc().unwrap()).collect();
        pages_tab.upload(&ids.iter().map(|&p| pages.addr(p)).collect::<Vec<u64>>());
        let state = LayerState::Attn {
            k: DeviceBuf::new(0),
            v: DeviceBuf::new(0),
            kr: DeviceBuf::new(IDX_RING * cfg.idx_dim * 4),
            kp: DeviceBuf::new(max_ctx.div_ceil(cfg.kpool) * cfg.idx_dim * 4),
            fp8: Some(Fp8Kv::new(1, window, nkv, hd, |_| pages_tab.ptr, pages.offsets(0))),
        };
        bytes.set(bytes.get() + 2 * window * nkv * (hd + 4) + max_ctx.div_ceil(cfg.kpool) * cfg.idx_dim * 4 + WMAX * HC * 3 * E * 4);
        let seg = |m: &Mat, y: &DeviceBuf| Seg { w: m.buf.ptr, y: y.ptr, rows: m.rows, cols: m.cols, qx: m.qx, stride: m.stride };
        let hc_groups = |h: &HcW| -> (MGemv, MGemv) {
            let mut first = vec![seg(&h.down, &s.lo)];
            if let Some(i) = &h.inject {
                first.push(seg(i, &s.inject));
            }
            (MGemv::new(&first), MGemv::new(&[seg(&h.up, &s.gate)]))
        };
        let groups = Groups {
            hc_attn: hc_groups(&hc_attn),
            hc_ffn: hc_groups(&hc_ffn),
            mixer_in: MGemv::new(&[seg(&attn.q, &s.qg), seg(&attn.k, &s.kraw), seg(&attn.v, &s.vnew), seg(&attn.ik, &s.kr), seg(&attn.iq, &s.iq)]),
            mixer_out: MGemv::new(&[seg(&attn.o, &s.y)]),
            moe_in: MGemv::new(&[seg(&router, &s.router), seg(&sh_gate, &s.sh_g), seg(&sh_up, &s.sh_u), seg(&sh_router, &s.sg)]),
            sh_down: MGemv::new(&[seg(&sh_down, &s.sh_y)]),
            ple: None,
        };
        let head_hc = hc_groups(&hc_head);
        let r = DeviceBuf::new(WMAX * HCD * 4);
        let eh = (0..WMAX * HC / 8)
            .map(|c| MGemv::new(&[Seg { w: eh_proj.buf.ptr, y: r.ptr + (c * 8 * E * 4) as u64, rows: eh_proj.rows, cols: eh_proj.cols, qx: eh_proj.qx, stride: eh_proj.stride }]))
            .collect();
        Ok(Self {
            hc_attn,
            hc_ffn,
            attn,
            eh_norm_e,
            eh_norm_h,
            hc_head,
            experts,
            rec_stride: rec,
            gate_bytes,
            gate_row,
            down_row,
            state,
            groups,
            head_hc,
            eh,
            r,
            cat: DeviceBuf::new(WMAX * HC * 2 * E * 4),
            pos: DeviceBuf::new(WMAX * 4),
            slot0: DeviceBuf::new(TMAX * 4),
            _pages: (pages, pages_tab),
            out: DeviceBuf::new(WMAX * 4),
            prob: DeviceBuf::new(WMAX * 4),
            bytes: bytes.get(),
            _mats: vec![router, sh_gate, sh_up, sh_down, sh_router, eh_proj],
            head: None,
            vocab: std::env::var("QX_DRAFT_VOCAB").ok().and_then(|s| s.parse().ok()).unwrap_or(32768).min(cfg.n_vocab),
            k: std::env::var("QX_SPEC_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4).clamp(1, WMAX - 1),
            graphs: (0..=WMAX).map(|_| None).collect(),
        })
    }
}

impl Drafter {
    /// Clears the drafter's sequence state (its KV is overwritten as positions are written).
    pub fn reset(&self) {
        if let LayerState::Attn { kr, kp, .. } = &self.state {
            kr.zero();
            kp.zero();
        }
    }

    /// The draft head over the trunk's output matrix.
    pub fn set_head(&mut self, output: &Mat, logits: &DeviceBuf) {
        self.head = Some(MGemv::new(&[Seg { w: output.buf.ptr, y: logits.ptr, rows: self.vocab, cols: output.cols, qx: output.qx, stride: output.stride }]));
    }
}

impl<'a> Engine<'a> {
    /// Sets the drafter's first position to the step's position plus `off`, on the device (graph-safe).
    pub(super) fn draft_pos(&self, off: i64) {
        let d = self.drafter.as_ref().unwrap();
        self.k("qx_draft_pos", 1, (1, 1, 1), &[self.step.ptr + (self.cur * STEP_ROW * 4) as u64, d.pos.ptr, off as u64]);
    }

    /// The drafter over t tokens at positions pos .. pos + t - 1 (its pos word): `toks` (device u32 x t) with the
    /// residuals `h` (device, t x HCD) of the positions before. Writes its KV. With `head`, the last token's
    /// proposal for the position after it goes to drafts[`slot`] with its probability.
    pub(super) fn draft_forward(&self, t: usize, h: u64, toks: u64, head: Option<usize>) {
        let d = self.drafter.as_ref().unwrap();
        let (c, s) = (&self.cfg, &self.s);
        ops::dequant_gather(&self.gpu, self.embd.ty, self.embd.buf.ptr, s.e.ptr, t, self.embd.cols, toks);
        self.k("qx_mtp_cat", 256, (HC as u32, t as u32, 1), &[s.e.ptr, h, d.eh_norm_e.ptr, d.eh_norm_h.ptr, d.cat.ptr, E as u64, bits(EPS)]);
        for (ci, m) in d.eh.iter().enumerate().take((t * HC).div_ceil(8)) {
            m.run_t(&self.gpu, d.cat.ptr + (ci * 8 * 2 * E * 4) as u64, 0, 0, 8, 2 * E);
        }
        let r = d.r.ptr;
        self.mix(r, &d.hc_attn, &d.groups.hc_attn, false, t);
        self.attn(&d.attn, &d.state, &d.groups, t, d.pos.ptr, d.slot0.ptr);
        self.combine(r, &d.hc_ffn.norm, t);
        self.mix(r, &d.hc_ffn, &d.groups.hc_ffn, true, t);
        // MoE: every expert resident
        let (ff, k) = (c.ff_exp, c.n_expert_used);
        let a = (t * k) as u32;
        d.groups.moe_in.run_t(&self.gpu, s.x.ptr, 0, 0, tpad(t), E);
        self.gpu.run(MOE.kernel("qx_moe_route", CubeDim::new_1d(32)), (t as u32, 1, 1), &[s.router.ptr, c.n_expert as u64, k as u64, s.ids.ptr, s.wts.ptr, 0, 0, 0, 0]);
        self.gpu.run(MOE.kernel("qx_moe_slots_all", CubeDim::new_1d(128)), (a.div_ceil(128), 1, 1), &[s.ids.ptr, s.slots.ptr, d.experts.ptr, d.rec_stride as u64, a as u64]);
        self.gpu.run(MOE.kernel("qx_moe_gateup_iq4_nl", CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), a, 1), &[s.slots.ptr, k as u64, 0, s.x.ptr, s.gu.ptr, ff as u64, E as u64, d.gate_row as u64]);
        self.gpu.run(MOE.kernel("qx_moe_down_iq4_nl", CubeDim::new_1d(256)), ((E as u32).div_ceil(8), a, 1), &[s.slots.ptr, 0, 0, 0, s.gu.ptr, s.yk.ptr, ff as u64, E as u64, 2 * d.gate_bytes as u64, d.down_row as u64]);
        d.groups.sh_down.run_t(&self.gpu, s.sh_g.ptr, 2, s.sh_u.ptr, tpad(t), ff);
        self.gpu.run(MOE.kernel("qx_moe_combine", CubeDim::new_1d(256)), (E.div_ceil(256) as u32, t as u32, 1), &[s.yk.ptr, k as u64, s.yk.ptr, s.y.ptr, E as u64, s.slots.ptr, s.wts.ptr, s.sh_y.ptr, s.sg.ptr]);
        if let Some(slot) = head {
            self.combine(r, &d.hc_head.norm, t);
            self.mix(r, &d.hc_head, &d.head_hc, true, t);
            // only the last token's proposal is wanted: the head runs on its row alone
            let n = d.vocab;
            d.head.as_ref().unwrap().run_t(&self.gpu, s.x.ptr + ((t - 1) * E * 4) as u64, 0, 0, 1, E);
            self.k("qx_argmax_p", 1024, (1, 1, 1), &[s.logits.ptr, n as u64, d.out.ptr + (slot * 4) as u64, d.prob.ptr + (slot * 4) as u64]);
        } else {
            self.combine(r, &d.hc_head.norm, t);
        }
    }

    /// The drafter's residual row of its last token (the chained step's h).
    fn draft_h(&self, row: usize) -> u64 {
        self.drafter.as_ref().unwrap().r.ptr + (row * HCD * 4) as u64
    }

    /// Catch-up after trunk tokens at positions p0 .. p0 + n - 1 whose residuals are `h` (n x HCD): the drafter
    /// takes the tokens at p0 + 1 .. p0 + n (`toks`, device) with them, in windows of WMAX.
    /// `step_pos` is where the step state's position is (the drafter's positions are set relative to it).
    pub(super) fn draft_catch_up(&self, p0: usize, n: usize, h: u64, toks: u64, head_last: bool, step_pos: usize) {
        let mut i = 0;
        while i < n {
            let t = (n - i).min(WMAX);
            self.draft_pos((p0 + 1 + i) as i64 - step_pos as i64);
            let last = i + t == n;
            self.draft_forward(t, h + (i * HCD * 4) as u64, toks + (i * 4) as u64, if last && head_last { Some(0) } else { None });
            i += t;
        }
    }

    /// The draft pass after a window that kept `keep` tokens (the window's samples, at positions pos - keep + 1
    /// .. pos where pos is the step state's): catch-up with the trunk's residuals, then k - 1 chained steps on
    /// the drafter's own residual. Positions come from the step state, so the pass is graph-safe.
    fn draft_pass(&self, keep: usize) {
        let d = self.drafter.as_ref().unwrap();
        let s = &self.s;
        self.draft_pos(1 - keep as i64);
        self.draft_forward(keep, s.r.ptr, s.samples.ptr, Some(0));
        let mut h_row = keep - 1;
        for j in 1..d.k {
            self.draft_pos(j as i64);
            self.draft_forward(1, self.draft_h(h_row), d.out.ptr + ((j - 1) * 4) as u64, Some(j));
            h_row = 0;
        }
    }

    /// Captures the draft pass for every catch-up width on a throwaway sequence (each capture needs an executed
    /// warm-up run first).
    pub(super) fn capture_drafts(&mut self) {
        let mut graphs: Vec<Option<cubecl_runtime::client::Graph>> = (0..=WMAX).map(|_| None).collect();
        // positions relative to the step's must stay in range during the warm-up runs
        self.set_step(2 * WMAX, 0, 0);
        for (t, slot) in graphs.iter_mut().enumerate().skip(1) {
            self.gpu.client.graph_prepare().expect("graph_prepare");
            self.draft_pass(t);
            self.gpu.client.start_capture().expect("start_capture");
            self.draft_pass(t);
            *slot = Some(self.gpu.client.stop_capture().expect("stop_capture"));
        }
        self.gpu.sync();
        self.drafter.as_mut().unwrap().graphs = graphs;
    }

    /// The drafts (and probabilities) for the next window, after one that kept `keep` tokens.
    pub(super) fn mtp_drafts(&self, keep: usize) -> (Vec<u32>, Vec<f32>) {
        let d = self.drafter.as_ref().unwrap();
        match &d.graphs[keep] {
            // SAFETY: every buffer the graph uses is owned by the engine and outlives it.
            Some(g) => unsafe { g.replay().expect("graph replay") },
            None => self.draft_pass(keep),
        }
        self.gpu.sync();
        let toks: Vec<u32> = d.out.download(d.k);
        let probs: Vec<f32> = d.prob.download(d.k);
        (toks, probs)
    }
}
