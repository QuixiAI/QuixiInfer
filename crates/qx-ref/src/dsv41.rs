//! The whole DeepSeek-V4.1-Flash forward pass on the CPU in f32, one token at a time: the reference the DeepSeek
//! engine's kernels are checked against. It follows the official `inference/model.py` and `engram.py` as the
//! llama.cpp PR ggml-org#30817 (`src/models/deepseek41.cpp`) renders them, including that port's activation names,
//! so `qx-ref check` can compare every intermediate with the oracle's golden files. Nothing here is fast beyond
//! parallel matvecs.
//!
//! Per layer, with the residual as `hc` = 4 copies R[c][n_embd]:
//! 1. Engram (two layers): 24 table rows hashed from the last 4 compressed token ids, `engram_kv` gives 4 keys
//!    and a value, a gate per copy from the key's cosine with the copy, R[c] += gate[c] * value.
//! 2. Hyper-connections (single-pass mHC): the mixes of one sublayer give the next sublayer's pre-mix weights,
//!    its own post weights and a 4x4 Sinkhorn-normalized combination matrix.
//! 3. Attention: 64 heads of 512 on one shared 512-wide latent (K = V), RoPE on the last 64 dims, a 128-token
//!    window plus, on compressed layers, the top-512 compressed latents picked by the lightning indexer (with
//!    candidate blocks from layer 20 for the index sources after it); per-head sinks; the output un-RoPEd, then
//!    grouped low-rank output projections.
//! 4. MoE: sqrt(softplus) router scores, selection by score + bias, 6 of 384 experts with clamped SwiGLU, plus
//!    one shared expert.
use crate::weights::{dot, W};
use anyhow::{anyhow, Result};
use qx_pack::gguf::Gguf;

const EPS: f32 = 1e-20;

pub struct Config {
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub hc: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub n_swa: usize,
    pub rope_base: f32,
    pub comp_rope_base: f32,
    pub yarn_factor: f32,
    pub yarn_orig_ctx: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_exp: usize,
    pub w_scale: f32,
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    pub ratio: Vec<usize>,
    /// Per layer, the layer whose compressed rows it reads (-1: window only) and the layer whose top-k it uses.
    pub kv_src: Vec<i32>,
    pub topk_src: Vec<i32>,
    pub cand_src: i32,
    pub cand_blocks: usize,
    pub cand_bs: usize,
    pub engram_layers: Vec<usize>,
    pub engram_ngram: usize,
    pub engram_heads: usize,
    pub engram_dim: usize,
    pub engram_mult: Vec<u64>,
    pub engram_offsets: Vec<u64>,
    pub engram_primes: Vec<u64>,
    pub engram_pad: u32,
    pub engram_token_map: Vec<u32>,
    /// The hyper-connection epsilon (`hyper_connection.epsilon`, 1e-6), distinct from the RMS epsilon.
    pub hc_eps: f32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let a = "deepseek41";
        let k = |s: &str| format!("{a}.{s}");
        let arr = |s: &str| -> Result<Vec<u64>> {
            g.get(&k(s)).and_then(|v| v.as_arr()).ok_or_else(|| anyhow!("missing {s}"))?.iter().map(|v| v.as_u64().ok_or_else(|| anyhow!("{s}"))).collect()
        };
        let farr = |s: &str| -> Result<Vec<f32>> {
            g.get(&k(s)).and_then(|v| v.as_arr()).ok_or_else(|| anyhow!("missing {s}"))?.iter().map(|v| v.as_f64().map(|x| x as f32).ok_or_else(|| anyhow!("{s}"))).collect()
        };
        let f = |s: &str| -> Result<f32> { Ok(g.get(&k(s)).and_then(|v| v.as_f64()).ok_or_else(|| anyhow!("missing {s}"))? as f32) };
        let n_layer = g.u64(&k("block_count"))? as usize;
        let ratio: Vec<usize> = arr("attention.compress_ratios")?.into_iter().take(n_layer).map(|v| v as usize).collect();
        let kv_src_layers = arr("attention.kv_source_layers")?;
        let idx_src_layers = arr("attention.indexer.source_layers")?;
        let (mut kv_src, mut topk_src) = (vec![-1i32; n_layer], vec![-1i32; n_layer]);
        let (mut last_kv, mut last_idx) = (-1i32, -1i32);
        for l in 0..n_layer {
            if kv_src_layers.contains(&(l as u64)) {
                last_kv = l as i32;
            }
            if idx_src_layers.contains(&(l as u64)) {
                last_idx = l as i32;
            }
            if ratio[l] != 0 {
                kv_src[l] = last_kv;
                topk_src[l] = last_idx;
            }
        }
        let clamp_exp = farr("swiglu_clamp_exp")?;
        let clamp_shexp = farr("swiglu_clamp_shexp").unwrap_or_else(|_| clamp_exp.clone());
        Ok(Self {
            n_layer,
            n_vocab: g.tensor("token_embd.weight").unwrap().shape[1] as usize,
            n_embd: g.u64(&k("embedding_length"))? as usize,
            hc: g.u64(&k("hyper_connection.count"))? as usize,
            n_head: g.u64(&k("attention.head_count"))? as usize,
            head_dim: g.u64(&k("attention.key_length"))? as usize,
            n_rot: g.u64(&k("rope.dimension_count"))? as usize,
            q_lora: g.u64(&k("attention.q_lora_rank"))? as usize,
            o_groups: g.u64(&k("attention.output_group_count"))? as usize,
            o_lora: g.u64(&k("attention.output_lora_rank"))? as usize,
            n_swa: g.u64(&k("attention.sliding_window"))? as usize,
            rope_base: f("rope.freq_base")?,
            comp_rope_base: f("attention.compress_rope_freq_base")?,
            yarn_factor: f("rope.scaling.factor")?,
            yarn_orig_ctx: g.u64(&k("rope.scaling.original_context_length"))? as usize,
            yarn_beta_fast: f("rope.scaling.yarn_beta_fast")?,
            yarn_beta_slow: f("rope.scaling.yarn_beta_slow")?,
            n_expert: g.u64(&k("expert_count"))? as usize,
            n_expert_used: g.u64(&k("expert_used_count"))? as usize,
            ff_exp: g.u64(&k("expert_feed_forward_length"))? as usize,
            w_scale: f("expert_weights_scale")?,
            clamp_exp,
            clamp_shexp,
            idx_heads: g.u64(&k("attention.indexer.head_count"))? as usize,
            idx_dim: g.u64(&k("attention.indexer.key_length"))? as usize,
            idx_top_k: g.u64(&k("attention.indexer.top_k"))? as usize,
            ratio,
            kv_src,
            topk_src,
            cand_src: g.u64(&k("attention.indexer.candidate_source_layer")).map(|v| v as i32).unwrap_or(-1),
            cand_blocks: g.u64(&k("attention.indexer.candidate_top_k_blocks")).unwrap_or(0) as usize,
            cand_bs: g.u64(&k("attention.indexer.candidate_block_size")).unwrap_or(0) as usize,
            engram_layers: arr("ple.layers")?.into_iter().map(|v| v as usize).collect(),
            engram_ngram: g.u64(&k("ple.ngram_size"))? as usize,
            engram_heads: g.u64(&k("ple.heads_per_ngram"))? as usize,
            engram_dim: g.u64(&k("embedding_length_per_layer_input"))? as usize,
            engram_mult: arr("ple.layer_multipliers")?,
            engram_offsets: arr("ple.head_offsets")?,
            engram_primes: arr("ple.head_vocab_sizes")?,
            engram_pad: g.u64(&k("engram.pad_token_id"))? as u32,
            engram_token_map: arr("engram.token_map")?.into_iter().map(|v| v as u32).collect(),
            hc_eps: f("hyper_connection.epsilon")?,
        })
    }

    /// Columns of the Engram table a token reads per Engram layer: one per (n-gram order, head).
    pub fn engram_cols(&self) -> usize {
        (self.engram_ngram - 1) * self.engram_heads
    }

    /// The Engram table rows of the token at `pos` of `tokens`, for Engram layer index `e`, in column order.
    pub fn engram_rows(&self, e: usize, tokens: &[u32], pos: usize) -> Vec<u64> {
        let ng = self.engram_ngram;
        let cols = self.engram_cols();
        let cid = |t: u32| self.engram_token_map.get(t as usize).map(|&c| c as u64);
        let pad = self.engram_token_map[self.engram_pad as usize] as u64;
        let mut toks = vec![pad; ng];
        let mut blocked = false;
        for s in 0..ng {
            let c = if pos >= s { cid(tokens[pos - s]) } else { None };
            blocked = blocked || c.is_none();
            toks[s] = if blocked { pad } else { c.unwrap() };
        }
        let mult = &self.engram_mult[e * ng..(e + 1) * ng];
        let primes = &self.engram_primes[e * cols..(e + 1) * cols];
        let offs = &self.engram_offsets[e * cols..(e + 1) * cols];
        let mut rows = vec![0u64; cols];
        let mut rolling = toks[0].wrapping_mul(mult[0]);
        for o in 1..ng {
            rolling ^= toks[o].wrapping_mul(mult[o]);
            for h in 0..self.engram_heads {
                let col = (o - 1) * self.engram_heads + h;
                rows[col] = rolling % primes[col] + offs[col];
            }
        }
        rows
    }
}

struct HcW<'a> {
    fn_: W<'a>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    sinks: Vec<f32>,
    wq_a: W<'a>,
    q_a_norm: Vec<f32>,
    wq_b: W<'a>,
    wkv: W<'a>,
    kv_norm: Vec<f32>,
    wo_a: W<'a>,
    wo_b: W<'a>,
    hc_attn: HcW<'a>,
    hc_ffn: HcW<'a>,
    comp_wkv: Option<W<'a>>,
    comp_wgate: Option<W<'a>>,
    comp_norm: Vec<f32>,
    idx_k: Option<W<'a>>,
    idx_k_norm: Vec<f32>,
    idx_q_b: Option<W<'a>>,
    idx_proj: Option<W<'a>>,
    engram_kv: Option<W<'a>>,
    engram_gate: Vec<f32>,
    router: W<'a>,
    router_bias: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: W<'a>,
    up: W<'a>,
    down: W<'a>,
    sh_gate: W<'a>,
    sh_up: W<'a>,
    sh_down: W<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    embd: W<'a>,
    output_norm: Vec<f32>,
    output: W<'a>,
    engram_table: Option<W<'a>>,
    layers: Vec<Layer<'a>>,
}

/// Per-sequence state.
pub struct State {
    pub pos: usize,
    pub tokens: Vec<u32>,
    /// Per layer, the latent of every position (`head_dim` each): the window reads the last `n_swa`.
    kv: Vec<Vec<f32>>,
    /// Per KV source layer, its compressed rows: `[comp (head_dim) | index key (idx_dim)]` per group.
    comp: Vec<Vec<f32>>,
    /// Per KV source layer, the pending group's (latent, score) pairs.
    pending: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
}

/// ggml's RMS scale: f32 squares summed in f64, the mean rounded to f32, `1 / sqrt(mean + eps)` in f32.
fn rms_scale(x: &[f32]) -> f32 {
    let sum: f64 = x.iter().map(|v| (v * v) as f64).sum();
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + EPS).sqrt()
}
fn rms_w(x: &[f32], w: &[f32], out: &mut [f32]) {
    let r = rms_scale(x);
    for i in 0..x.len() {
        out[i] = x[i] * r * w[i];
    }
}
fn rms(x: &[f32]) -> Vec<f32> {
    let r = rms_scale(x);
    x.iter().map(|v| v * r).collect()
}
fn rmsw(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    rms_w(x, w, &mut o);
    o
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}
/// The `k` largest, by value descending then index ascending.
fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// ggml's RoPE (NORM mode: consecutive pairs) with YaRN, on `n_rot` dims starting at `off`.
struct Rope {
    n_rot: usize,
    base: f32,
    freq_scale: f32,
    ext: bool,
    corr: [f32; 2],
}

impl Rope {
    fn new(c: &Config, compressed: bool) -> Self {
        let n = c.n_rot;
        if compressed {
            let base = c.comp_rope_base;
            let dim = |n_rot: f32| n as f32 * (c.yarn_orig_ctx as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
            let corr = [dim(c.yarn_beta_fast).floor().max(0.0), dim(c.yarn_beta_slow).ceil().min(n as f32 - 1.0)];
            Self { n_rot: n, base, freq_scale: 1.0 / c.yarn_factor, ext: true, corr }
        } else {
            Self { n_rot: n, base: c.rope_base, freq_scale: 1.0, ext: false, corr: [0.0, 0.0] }
        }
    }

    fn apply(&self, x: &mut [f32], off: usize, pos: usize, back: bool) {
        let n = self.n_rot;
        let mut theta_base = pos as f32;
        let theta_scale = self.base.powf(-2.0 / n as f32);
        for i0 in (0..n).step_by(2) {
            let theta_extrap = theta_base;
            let theta_interp = self.freq_scale * theta_extrap;
            let mut theta = theta_interp;
            // mscale: attn_factor 1/(1 + 0.1 ln(1/freq_scale)) times YaRN's 1 + 0.1 ln(1/freq_scale) = 1
            if self.ext {
                let y = ((i0 / 2) as f32 - self.corr[0]) / (self.corr[1] - self.corr[0]).max(0.001);
                let ramp = 1.0 - y.clamp(0.0, 1.0);
                theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
            }
            let (s, c) = theta.sin_cos();
            let s = if back { -s } else { s };
            let (x0, x1) = (x[off + i0], x[off + i0 + 1]);
            x[off + i0] = x0 * c - x1 * s;
            x[off + i0 + 1] = x0 * s + x1 * c;
            theta_base *= theta_scale;
        }
    }
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let w = |n: &str| -> Result<W<'a>> { Ok(W::new(g, g.tensor(n).ok_or_else(|| anyhow!("missing tensor {n}"))?)) };
        let wo = |n: &str| -> Result<Option<W<'a>>> { Ok(g.tensor(n).map(|t| W::new(g, t))) };
        let v = |n: &str| -> Result<Vec<f32>> { Ok(w(n)?.to_vec()) };
        let vo = |n: &str| -> Vec<f32> { g.tensor(n).map(|t| W::new(g, t).to_vec()).unwrap_or_default() };
        let hcw = |p: &str| -> Result<HcW<'a>> { Ok(HcW { fn_: w(&format!("{p}_fn.weight"))?, base: v(&format!("{p}_base.weight"))?, scale: v(&format!("{p}_scale.weight"))? }) };
        let mut layers = Vec::new();
        for l in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{l}.{s}");
            layers.push(Layer {
                attn_norm: v(&b("attn_norm.weight"))?,
                sinks: v(&b("attn_sinks.weight"))?,
                wq_a: w(&b("attn_q_a.weight"))?,
                q_a_norm: v(&b("attn_q_a_norm.weight"))?,
                wq_b: w(&b("attn_q_b.weight"))?,
                wkv: w(&b("attn_kv.weight"))?,
                kv_norm: v(&b("attn_kv_a_norm.weight"))?,
                wo_a: w(&b("attn_output_a.weight"))?,
                wo_b: w(&b("attn_output_b.weight"))?,
                hc_attn: hcw(&b("hc_attn"))?,
                hc_ffn: hcw(&b("hc_ffn"))?,
                comp_wkv: wo(&b("attn_compressor_kv.weight"))?,
                comp_wgate: wo(&b("attn_compressor_gate.weight"))?,
                comp_norm: vo(&b("attn_compressor_norm.weight")),
                idx_k: wo(&b("indexer.attn_k.weight"))?,
                idx_k_norm: vo(&b("indexer.k_norm.weight")),
                idx_q_b: wo(&b("indexer.attn_q_b.weight"))?,
                idx_proj: wo(&b("indexer.proj.weight"))?,
                engram_kv: wo(&b("engram_kv.weight"))?,
                engram_gate: vo(&b("engram_gate.weight")),
                router: w(&b("ffn_gate_inp.weight"))?,
                router_bias: v(&b("exp_probs_b.bias"))?,
                ffn_norm: v(&b("ffn_norm.weight"))?,
                gate: w(&b("ffn_gate_exps.weight"))?,
                up: w(&b("ffn_up_exps.weight"))?,
                down: w(&b("ffn_down_exps.weight"))?,
                sh_gate: w(&b("ffn_gate_shexp.weight"))?,
                sh_up: w(&b("ffn_up_shexp.weight"))?,
                sh_down: w(&b("ffn_down_shexp.weight"))?,
            });
        }
        Ok(Self { embd: w("token_embd.weight")?, output_norm: v("output_norm.weight")?, output: w("output.weight")?, engram_table: wo("per_layer_token_embd.weight")?, layers, cfg })
    }

    pub fn new_state(&self) -> State {
        let n = self.cfg.n_layer;
        State { pos: 0, tokens: Vec::new(), kv: vec![Vec::new(); n], comp: vec![Vec::new(); n], pending: vec![Vec::new(); n] }
    }

    /// The hyper-connection mixes of a sublayer: `(pre, post, comb)` with `comb[src * hc + dst]`.
    fn hc_mixes(&self, w: &HcW, r: &[f32], act: &mut dyn FnMut(&str, &[f32]), l: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hc = self.cfg.hc;
        let eps = self.cfg.hc_eps;
        let mixes = w.fn_.matvec(&rms(r));
        act(&format!("hc_mixes-{l}"), &mixes);
        let pre: Vec<f32> = (0..hc).map(|i| sigmoid(mixes[i] * w.scale[0] + w.base[i]) + eps).collect();
        act(&format!("hc_pre-{l}"), &pre);
        let post: Vec<f32> = (0..hc).map(|i| 2.0 * sigmoid(mixes[hc + i] * w.scale[1] + w.base[hc + i])).collect();
        act(&format!("hc_post-{l}"), &post);
        // comb[dst fastest, src]: softmax over dst per src, then Sinkhorn column/row normalizations
        let mut comb = vec![0f32; hc * hc];
        for s in 0..hc {
            let raw: Vec<f32> = (0..hc).map(|d| mixes[2 * hc + s * hc + d] * w.scale[2] + w.base[2 * hc + s * hc + d]).collect();
            let sm = softmax(&raw);
            for d in 0..hc {
                comb[s * hc + d] = sm[d] + eps;
            }
        }
        let norm_cols = |c: &mut [f32]| {
            for d in 0..hc {
                let sum: f32 = (0..hc).map(|s| c[s * hc + d]).sum::<f32>() + eps;
                for s in 0..hc {
                    c[s * hc + d] /= sum;
                }
            }
        };
        let norm_rows = |c: &mut [f32]| {
            for s in 0..hc {
                let sum: f32 = c[s * hc..(s + 1) * hc].iter().sum::<f32>() + eps;
                for d in 0..hc {
                    c[s * hc + d] /= sum;
                }
            }
        };
        norm_cols(&mut comb);
        for _ in 1..20 {
            norm_rows(&mut comb);
            norm_cols(&mut comb);
        }
        act(&format!("hc_comb-{l}"), &comb);
        (pre, post, comb)
    }

    fn hc_pre(&self, r: &[f32], pre: &[f32]) -> Vec<f32> {
        let (e, hc) = (self.cfg.n_embd, self.cfg.hc);
        let mut x = vec![0f32; e];
        for c in 0..hc {
            for i in 0..e {
                x[i] += r[c * e + i] * pre[c];
            }
        }
        x
    }

    fn hc_post(&self, y: &[f32], r: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
        let (e, hc) = (self.cfg.n_embd, self.cfg.hc);
        let mut out = vec![0f32; hc * e];
        for d in 0..hc {
            for i in 0..e {
                let mut v = y[i] * post[d];
                for s in 0..hc {
                    v += r[s * e + i] * comb[s * hc + d];
                }
                out[d * e + i] = v;
            }
        }
        out
    }

    fn engram(&self, w: &Layer, e: usize, r: &mut [f32], st: &State, l: usize, act: &mut dyn FnMut(&str, &[f32])) {
        let c = &self.cfg;
        let (ne, hc, ed) = (c.n_embd, c.hc, c.engram_dim);
        let rows = c.engram_rows(e, &st.tokens, st.pos);
        let table = self.engram_table.as_ref().expect("engram table");
        let mut emb = vec![0f32; rows.len() * ed];
        for (i, &row) in rows.iter().enumerate() {
            table.row(row as usize, &mut emb[i * ed..(i + 1) * ed]);
        }
        act(&format!("engram_rows-{l}"), &emb);
        let kv = w.engram_kv.as_ref().unwrap().matvec(&emb);
        act(&format!("engram_kv-{l}"), &kv);
        let value = &kv[hc * ne..(hc + 1) * ne];
        let mut dots = vec![0f32; hc];
        for ci in 0..hc {
            let xn = rms(&r[ci * ne..(ci + 1) * ne]);
            let kn = rms(&kv[ci * ne..(ci + 1) * ne]);
            let g = &w.engram_gate[ci * ne..(ci + 1) * ne];
            dots[ci] = (0..ne).map(|i| xn[i] * kn[i] * g[i]).sum::<f32>() / (ne as f32).sqrt();
        }
        act(&format!("engram_dot-{l}"), &dots);
        let gates: Vec<f32> = dots.iter().map(|&d| sigmoid(d.abs().max(1e-6).sqrt() * if d < 0.0 { -1.0 } else { 1.0 })).collect();
        act(&format!("engram_gate-{l}"), &gates);
        for ci in 0..hc {
            for i in 0..ne {
                r[ci * ne + i] += value[i] * gates[ci];
            }
        }
    }

    fn attention(&self, w: &Layer, x: &[f32], l: usize, st: &mut State, topk_of: &mut Vec<Option<Vec<usize>>>, cand_sel: &mut Option<Vec<usize>>, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let c = &self.cfg;
        let (nh, hd, nr) = (c.n_head, c.head_dim, c.n_rot);
        let nope = hd - nr;
        let pos = st.pos;
        let ratio = c.ratio[l];
        let rope = Rope::new(c, ratio != 0);

        let qr = rmsw(&w.wq_a.matvec(x), &w.q_a_norm);
        act(&format!("qr_norm-{l}"), &qr);
        let mut q = w.wq_b.matvec(&qr);
        for h in 0..nh {
            rope.apply(&mut q[h * hd..(h + 1) * hd], nope, pos, false);
        }
        act(&format!("q-{l}"), &q);
        let mut kv = rmsw(&w.wkv.matvec(x), &w.kv_norm);
        rope.apply(&mut kv, nope, pos, false);
        act(&format!("kv-{l}"), &kv);
        st.kv[l].extend_from_slice(&kv);

        // the compressor of a KV source layer: a group's latents pooled by a per-dimension softmax gate
        let mut selected: Vec<usize> = Vec::new();
        if ratio != 0 {
            let src = c.kv_src[l] as usize;
            if src == l {
                let state_kv = w.comp_wkv.as_ref().unwrap().matvec(x);
                let score = match &w.comp_wgate {
                    Some(g) if ratio > 1 => g.matvec(x),
                    _ => vec![0f32; hd],
                };
                st.pending[l].push((state_kv, score));
                if (pos + 1) % ratio == 0 {
                    let group: Vec<(Vec<f32>, Vec<f32>)> = st.pending[l].drain(..).collect();
                    let mut comp = vec![0f32; hd];
                    for i in 0..hd {
                        let sc: Vec<f32> = group.iter().map(|(_, s)| s[i]).collect();
                        let sm = softmax(&sc);
                        comp[i] = group.iter().zip(&sm).map(|((v, _), p)| v[i] * p).sum();
                    }
                    let comp = rmsw(&comp, &w.comp_norm);
                    act(&format!("comp_latent-{l}"), &comp);
                    let gpos = pos + 1 - ratio;
                    let mut ik = rmsw(&w.idx_k.as_ref().unwrap().matvec(&comp), &w.idx_k_norm);
                    rope.apply(&mut ik, c.idx_dim - nr, gpos, false);
                    let mut comp = comp;
                    rope.apply(&mut comp, nope, gpos, false);
                    st.comp[l].extend_from_slice(&comp);
                    st.comp[l].extend_from_slice(&ik);
                }
            }
            let row = hd + c.idx_dim;
            let n_visible = (pos + 1) / ratio;
            assert_eq!(st.comp[src].len() / row, n_visible, "layer {l}: compressed rows of source {src}");
            if n_visible > 0 {
                let tsrc = c.topk_src[l] as usize;
                if tsrc == l {
                    // lightning indexer: relu(q_h . k) weighted per head, over the visible compressed rows
                    let mut iq = w.idx_q_b.as_ref().unwrap().matvec(&qr);
                    for h in 0..c.idx_heads {
                        rope.apply(&mut iq[h * c.idx_dim..(h + 1) * c.idx_dim], c.idx_dim - nr, pos, false);
                    }
                    let iw: Vec<f32> = w.idx_proj.as_ref().unwrap().matvec(x).into_iter().map(|v| v / ((c.idx_dim * c.idx_heads) as f32).sqrt()).collect();
                    let mut score = vec![0f32; n_visible];
                    for r in 0..n_visible {
                        let k = &st.comp[src][r * row + hd..(r + 1) * row];
                        score[r] = (0..c.idx_heads).map(|h| dot(&iq[h * c.idx_dim..(h + 1) * c.idx_dim], k).max(0.0) * iw[h]).sum();
                    }
                    act(&format!("idx_score-{l}"), &score);
                    let bs = c.cand_bs;
                    if c.cand_src == l as i32 {
                        // level one: the best position per block, the newest block pinned, the top blocks kept
                        let n_blk = n_visible.div_ceil(bs);
                        let mut bmax = vec![f32::NEG_INFINITY; n_blk];
                        for (r, &s) in score.iter().enumerate() {
                            bmax[r / bs] = bmax[r / bs].max(s);
                        }
                        bmax[(n_visible - 1) / bs] += 1e30;
                        act(&format!("cand_block_score-{l}"), &bmax);
                        *cand_sel = Some(top_k(&bmax, c.cand_blocks.min(n_blk)));
                    } else if c.cand_src >= 0 && (c.cand_src as usize) < l {
                        // level two: only positions inside the source's candidate blocks
                        let sel = cand_sel.as_ref().expect("candidate blocks");
                        let mut inside = vec![false; n_visible.div_ceil(bs)];
                        for &b in sel {
                            if b < inside.len() {
                                inside[b] = true;
                            }
                        }
                        for (r, s) in score.iter_mut().enumerate() {
                            if !inside[r / bs] {
                                *s = f32::NEG_INFINITY;
                            }
                        }
                    }
                    topk_of[l] = Some(top_k(&score, c.idx_top_k.min(n_visible)));
                }
                selected = topk_of[tsrc].clone().expect("top-k source");
            }
        }

        // attention over [window | selected compressed latents], K = V, per-head sinks
        let first = (pos + 1).saturating_sub(c.n_swa);
        let keys: Vec<&[f32]> = (first..=pos).map(|p| &st.kv[l][p * hd..(p + 1) * hd]).chain(selected.iter().map(|&r| {
            let src = c.kv_src[l] as usize;
            let row = hd + c.idx_dim;
            &st.comp[src][r * row..r * row + hd]
        })).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut o = vec![0f32; nh * hd];
        for h in 0..nh {
            let qh = &q[h * hd..(h + 1) * hd];
            let sc: Vec<f32> = keys.iter().map(|k| dot(qh, k) * scale).collect();
            let m = sc.iter().cloned().fold(w.sinks[h], f32::max);
            let p: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = p.iter().sum::<f32>() + (w.sinks[h] - m).exp();
            let oh = &mut o[h * hd..(h + 1) * hd];
            for (pi, k) in p.iter().zip(&keys) {
                for i in 0..hd {
                    oh[i] += pi / z * k[i];
                }
            }
        }
        act(&format!("{}-{l}", if ratio == 0 || selected.is_empty() { "attn_raw" } else { "attn_comp" }), &o);
        for h in 0..nh {
            rope.apply(&mut o[h * hd..(h + 1) * hd], nope, pos, true);
        }
        act(&format!("attn_derope-{l}"), &o);
        let gd = nh / c.o_groups * hd;
        let mut oa = vec![0f32; c.o_groups * c.o_lora];
        for g in 0..c.o_groups {
            w.wo_a.matvec_rows(g * c.o_lora, &o[g * gd..(g + 1) * gd], &mut oa[g * c.o_lora..(g + 1) * c.o_lora]);
        }
        act(&format!("attn_wo_a-{l}"), &oa);
        let out = w.wo_b.matvec(&oa);
        act(&format!("attn_out-{l}"), &out);
        out
    }

    fn moe(&self, w: &Layer, x: &[f32], l: usize, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let c = &self.cfg;
        let (ne, ff) = (c.n_embd, c.ff_exp);
        let logits = w.router.matvec(x);
        act(&format!("ffn_moe_logits-{l}"), &logits);
        let probs: Vec<f32> = logits.iter().map(|&v| softplus(v).sqrt()).collect();
        let sel: Vec<f32> = probs.iter().zip(&w.router_bias).map(|(p, b)| p + b).collect();
        let idx = top_k(&sel, c.n_expert_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>() + 1e-20;
        let limit = c.clamp_exp[l];
        let mut y = vec![0f32; ne];
        for &e in &idx {
            let (mut g, mut u) = (vec![0f32; ff], vec![0f32; ff]);
            w.gate.matvec_rows_expert(e * ff, x, &mut g);
            w.up.matvec_rows_expert(e * ff, x, &mut u);
            let h: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a.min(limit)) * b.clamp(-limit, limit)).collect();
            let mut d = vec![0f32; ne];
            w.down.matvec_rows_expert_down(e * ne, &h, &mut d);
            let we = probs[e] / wsum * c.w_scale;
            for i in 0..ne {
                y[i] += we * d[i];
            }
        }
        act(&format!("ffn_moe_out-{l}"), &y);
        let limit = c.clamp_shexp[l];
        let g = w.sh_gate.matvec(x);
        let u = w.sh_up.matvec(x);
        let h: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a.min(limit)) * b.clamp(-limit, limit)).collect();
        let d = w.sh_down.matvec(&h);
        act(&format!("ffn_shexp-{l}"), &d);
        for i in 0..ne {
            y[i] += d[i];
        }
        y
    }

    /// Runs one token at position `st.pos`; returns the logits. `act` receives named activations.
    pub fn forward(&self, tok: u32, st: &mut State, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let c = &self.cfg;
        let (ne, hc) = (c.n_embd, c.hc);
        st.tokens.push(tok);
        let mut e = vec![0f32; ne];
        self.embd.row(tok as usize, &mut e);
        let mut r = vec![0f32; hc * ne];
        for ci in 0..hc {
            r[ci * ne..(ci + 1) * ne].copy_from_slice(&e);
        }
        act("hc_init", &r);
        let mut pre_prev: Option<Vec<f32>> = None;
        let mut topk_of: Vec<Option<Vec<usize>>> = vec![None; c.n_layer];
        let mut cand_sel: Option<Vec<usize>> = None;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(ei) = c.engram_layers.iter().position(|&x| x == l) {
                self.engram(layer, ei, &mut r, st, l, act);
                act(&format!("engram_out-{l}"), &r);
            }
            let (attn_pre, post, comb) = self.hc_mixes(&layer.hc_attn, &r, act, l);
            let x = match &pre_prev {
                Some(p) => self.hc_pre(&r, p),
                None => r[..ne].to_vec(),
            };
            act(&format!("hc_attn_pre-{l}"), &x);
            let x = rmsw(&x, &layer.attn_norm);
            act(&format!("attn_norm-{l}"), &x);
            let y = self.attention(layer, &x, l, st, &mut topk_of, &mut cand_sel, act);
            r = self.hc_post(&y, &r, &post, &comb);
            act(&format!("hc_attn_post-{l}"), &r);
            let (ffn_pre, post, comb) = self.hc_mixes(&layer.hc_ffn, &r, act, l);
            let x = self.hc_pre(&r, &attn_pre);
            act(&format!("hc_ffn_pre-{l}"), &x);
            let x = rmsw(&x, &layer.ffn_norm);
            act(&format!("ffn_norm-{l}"), &x);
            let y = self.moe(layer, &x, l, act);
            act(&format!("ffn_out-{l}"), &y);
            r = self.hc_post(&y, &r, &post, &comb);
            act(&format!("l_out-{l}"), &r);
            pre_prev = Some(ffn_pre);
        }
        let x = self.hc_pre(&r, pre_prev.as_ref().unwrap());
        act("hc_out", &x);
        let x = rmsw(&x, &self.output_norm);
        act("result_norm", &x);
        st.pos += 1;
        self.output.matvec(&x)
    }
}
