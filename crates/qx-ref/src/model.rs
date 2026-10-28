//! The whole Qwen3.8-Flash-Next forward pass on the CPU in f32, one token at a time.
//!
//! This is the reference QuixiInfer's GPU and CPU kernels are checked against. It follows
//! docs/model-forward.md line by line and makes no attempt to be fast beyond parallel matvecs.
use crate::weights::{dot, W};
use anyhow::{anyhow, Result};
use qx_pack::ggml::GgmlType;
use qx_pack::gguf::Gguf;

pub const E: usize = 2560;
pub const HC: usize = 4;
pub const HCD: usize = HC * E;
const EPS: f32 = 1e-6;

pub struct Config {
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_exp: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub ssm_dim: usize,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    pub kpool: usize,
    pub ple_layer: usize,
    pub ple_dim: usize,
    pub ple_heads_per_gram: usize,
    pub ple_ngram: usize,
    pub ple_mult: Vec<u64>,
    pub ple_offsets: Vec<u64>,
    pub ple_vocab: Vec<u64>,
    pub ple_conv_k: usize,
    pub eos: u32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let a = "qwen4exp";
        let k = |s: &str| format!("{a}.{s}");
        let arr = |s: &str| -> Result<Vec<u64>> {
            g.get(&k(s)).and_then(|v| v.as_arr()).ok_or_else(|| anyhow!("missing {s}"))?.iter().map(|v| v.as_u64().ok_or_else(|| anyhow!("{s}"))).collect()
        };
        let ple_layers = arr("ple.layers")?;
        Ok(Self {
            n_layer: g.u64(&k("block_count"))? as usize,
            n_vocab: g.tensor("token_embd.weight").unwrap().shape[1] as usize,
            n_head: g.u64(&k("attention.head_count"))? as usize,
            n_head_kv: g.u64(&k("attention.head_count_kv"))? as usize,
            head_dim: g.u64(&k("attention.key_length"))? as usize,
            n_rot: g.u64(&k("rope.dimension_count"))? as usize,
            rope_base: g.get(&k("rope.freq_base")).and_then(|v| v.as_f64()).unwrap() as f32,
            n_expert: g.u64(&k("expert_count"))? as usize,
            n_expert_used: g.u64(&k("expert_used_count"))? as usize,
            ff_exp: g.u64(&k("expert_feed_forward_length"))? as usize,
            ssm_k_heads: g.u64(&k("ssm.group_count"))? as usize,
            ssm_v_heads: g.u64(&k("ssm.time_step_rank"))? as usize,
            ssm_dim: g.u64(&k("ssm.state_size"))? as usize,
            idx_heads: g.u64(&k("attention.indexer.head_count"))? as usize,
            idx_dim: g.u64(&k("attention.indexer.key_length"))? as usize,
            idx_top_k: g.u64(&k("attention.indexer.top_k"))? as usize,
            kpool: 4,
            ple_layer: ple_layers[0] as usize,
            ple_dim: g.u64(&k("embedding_length_per_layer_input"))? as usize,
            ple_heads_per_gram: g.u64(&k("ple.heads_per_ngram"))? as usize,
            ple_ngram: g.u64(&k("ple.ngram_size"))? as usize,
            ple_mult: arr("ple.layer_multipliers")?,
            ple_offsets: arr("ple.head_offsets")?,
            ple_vocab: arr("ple.head_vocab_sizes")?,
            ple_conv_k: g.u64(&k("ple.conv_kernel"))? as usize,
            eos: g.u64(&k("ple.eos_token_id"))? as u32,
        })
    }
    pub fn is_attn(&self, l: usize) -> bool {
        (l + 1) % 4 == 0
    }
}

struct HcW<'a> {
    norm: Vec<f32>,
    down: W<'a>,
    up: W<'a>,
    inject: Option<W<'a>>,
}

struct Gdn<'a> {
    qkv: W<'a>,
    gate: W<'a>,
    beta: W<'a>,
    alpha: W<'a>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    conv: Vec<f32>,
    norm: Vec<f32>,
    out: W<'a>,
}

struct Attn<'a> {
    q: W<'a>,
    k: W<'a>,
    v: W<'a>,
    o: W<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    iq: W<'a>,
    ik: W<'a>,
    iq_norm: Vec<f32>,
    ik_norm: Vec<f32>,
}

struct Moe<'a> {
    router: W<'a>,
    gate: W<'a>,
    up: W<'a>,
    down: W<'a>,
    sh_gate: W<'a>,
    sh_up: W<'a>,
    sh_down: W<'a>,
    sh_router: Vec<f32>,
    /// NVFP4 experts: per-expert per-tensor scales of gate, up and down (ModelOpt's `weight_scale_2`, the GGUF's
    /// `ffn_*_exps.scale`); 1 where the tensor is absent.
    scale: Option<[Vec<f32>; 3]>,
}

struct Ple<'a> {
    table: W<'a>,
    key: W<'a>,
    value: W<'a>,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
}

enum Mixer<'a> {
    Gdn(Gdn<'a>),
    Attn(Attn<'a>),
}

struct Layer<'a> {
    hc_attn: HcW<'a>,
    hc_ffn: HcW<'a>,
    mixer: Mixer<'a>,
    moe: Moe<'a>,
    ple: Option<Ple<'a>>,
}

pub struct Model<'a> {
    pub cfg: Config,
    embd: W<'a>,
    output: W<'a>,
    head: HcW<'a>,
    layers: Vec<Layer<'a>>,
}

/// Per-sequence state.
pub struct State {
    pub pos: usize,
    pub tokens: Vec<u32>,
    /// GDN: per layer, conv window (3 x 10240, oldest first) and 48 states of 128 x 128 (`s[j * 128 + i] = S[i][j]`).
    conv: Vec<Vec<f32>>,
    ssm: Vec<Vec<f32>>,
    /// Attention: per layer, K and V per cell (2 heads x 256 each), raw indexer keys, pooled block keys.
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    kr: Vec<Vec<f32>>,
    kp: Vec<Vec<f32>>,
    /// PLE conv history: the last 9 normalised inputs (10240 each), oldest first.
    ple_hist: Vec<f32>,
}

/// The PLE table rows of the last token of `tokens` (one sequence from its start).
pub fn ple_rows(c: &Config, tokens: &[u32]) -> Vec<u64> {
    let n = tokens.len();
    let mut ctx = vec![c.eos as u64; c.ple_ngram];
    ctx[0] = tokens[n - 1] as u64;
    let mut cut = false;
    for s in 1..c.ple_ngram {
        let t = if n > s { Some(tokens[n - 1 - s]) } else { None };
        cut = cut || t.is_none() || t == Some(c.eos);
        ctx[s] = if cut { c.eos as u64 } else { t.unwrap() as u64 };
    }
    let mut rows = Vec::new();
    for ng in 2..=c.ple_ngram {
        let mut m = ctx[0].wrapping_mul(c.ple_mult[0]);
        for j in 1..ng {
            m ^= ctx[j].wrapping_mul(c.ple_mult[j]);
        }
        for g in 0..c.ple_heads_per_gram {
            let h = (ng - 2) * c.ple_heads_per_gram + g;
            rows.push(m % c.ple_vocab[h] + c.ple_offsets[h]);
        }
    }
    rows
}

fn rms(x: &[f32], w: &[f32], out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + EPS).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * r * w[i];
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let w = |n: &str| -> Result<W<'a>> { Ok(W::new(g, g.tensor(n).ok_or_else(|| anyhow!("missing tensor {n}"))?)) };
        let v = |n: &str| -> Result<Vec<f32>> { Ok(w(n)?.to_vec()) };
        let hc = |p: &str, inject: bool| -> Result<HcW<'a>> {
            Ok(HcW {
                norm: v(&format!("{p}_norm.weight"))?,
                down: w(&format!("{p}_down.weight"))?,
                up: w(&format!("{p}_up.weight"))?,
                inject: if inject { Some(w(&format!("{p}_inject.weight"))?) } else { None },
            })
        };
        let mut layers = Vec::new();
        for l in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{l}.{s}");
            let mixer = if cfg.is_attn(l) {
                Mixer::Attn(Attn {
                    q: w(&b("attn_q.weight"))?,
                    k: w(&b("attn_k.weight"))?,
                    v: w(&b("attn_v.weight"))?,
                    o: w(&b("attn_output.weight"))?,
                    q_norm: v(&b("attn_q_norm.weight"))?,
                    k_norm: v(&b("attn_k_norm.weight"))?,
                    iq: w(&b("indexer.q_proj.weight"))?,
                    ik: w(&b("indexer.k_proj.weight"))?,
                    iq_norm: v(&b("indexer.q_norm.weight"))?,
                    ik_norm: v(&b("indexer.k_norm.weight"))?,
                })
            } else {
                Mixer::Gdn(Gdn {
                    qkv: w(&b("attn_qkv.weight"))?,
                    gate: w(&b("attn_gate.weight"))?,
                    beta: w(&b("ssm_beta.weight"))?,
                    alpha: w(&b("ssm_alpha.weight"))?,
                    dt_bias: v(&b("ssm_dt.bias"))?,
                    a: v(&b("ssm_a"))?,
                    conv: v(&b("ssm_conv1d.weight"))?,
                    norm: v(&b("ssm_norm.weight"))?,
                    out: w(&b("ssm_out.weight"))?,
                })
            };
            let ple = if l == cfg.ple_layer {
                Some(Ple {
                    table: w("per_layer_token_embd.weight")?,
                    key: w(&b("ple_key.weight"))?,
                    value: w(&b("ple_value.weight"))?,
                    norm_key: v(&b("ple_norm_key.weight"))?,
                    norm_query: v(&b("ple_norm_query.weight"))?,
                    norm_conv: v(&b("ple_norm_conv.weight"))?,
                    conv: v(&b("ple_conv1d.weight"))?,
                })
            } else {
                None
            };
            layers.push(Layer {
                hc_attn: hc(&b("hc_attn"), true)?,
                hc_ffn: hc(&b("hc_ffn"), true)?,
                mixer,
                moe: Moe {
                    router: w(&b("ffn_gate_inp.weight"))?,
                    gate: w(&b("ffn_gate_exps.weight"))?,
                    up: w(&b("ffn_up_exps.weight"))?,
                    down: w(&b("ffn_down_exps.weight"))?,
                    sh_gate: w(&b("ffn_gate_shexp.weight"))?,
                    sh_up: w(&b("ffn_up_shexp.weight"))?,
                    sh_down: w(&b("ffn_down_shexp.weight"))?,
                    sh_router: v(&b("ffn_gate_inp_shexp.weight"))?,
                    scale: {
                        let sc = |n: &str| -> Result<Vec<f32>> { Ok(match g.tensor(&b(&format!("ffn_{n}_exps.scale"))) { Some(t) => W::new(g, t).to_vec(), None => vec![1.0; cfg.n_expert] }) };
                        (g.tensor(&b("ffn_gate_exps.weight")).map(|t| t.ty) == Some(GgmlType::NVFP4)).then(|| Ok::<_, anyhow::Error>([sc("gate")?, sc("up")?, sc("down")?])).transpose()?
                    },
                },
                ple,
            });
        }
        Ok(Self { embd: w("token_embd.weight")?, output: w("output.weight")?, head: hc("output_hc", false)?, layers, cfg })
    }

    pub fn new_state(&self) -> State {
        let c = &self.cfg;
        let n = c.n_layer;
        let mut st = State {
            pos: 0,
            tokens: Vec::new(),
            conv: vec![Vec::new(); n],
            ssm: vec![Vec::new(); n],
            kc: vec![Vec::new(); n],
            vc: vec![Vec::new(); n],
            kr: vec![Vec::new(); n],
            kp: vec![Vec::new(); n],
            ple_hist: vec![0f32; (c.ple_conv_k - 1) * c.ple_ngram * HCD],
        };
        for l in 0..n {
            if !c.is_attn(l) {
                st.conv[l] = vec![0f32; 3 * self.conv_channels()];
                st.ssm[l] = vec![0f32; c.ssm_v_heads * c.ssm_dim * c.ssm_dim];
            }
        }
        st
    }

    fn conv_channels(&self) -> usize {
        let c = &self.cfg;
        2 * c.ssm_k_heads * c.ssm_dim + c.ssm_v_heads * c.ssm_dim
    }

    /// Returns `(mixed, inject)`.
    fn mix(&self, w: &HcW, r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut xn = vec![0f32; HCD];
        for s in 0..HC {
            rms(&r[s * E..(s + 1) * E], &w.norm[s * E..(s + 1) * E], &mut xn[s * E..(s + 1) * E]);
        }
        let lo: Vec<f32> = w.down.matvec(&xn).into_iter().map(|v| silu(v / HC as f32)).collect();
        let gate = w.up.matvec(&lo);
        let mut mixed = vec![0f32; E];
        for s in 0..HC {
            for e in 0..E {
                mixed[e] += xn[s * E + e] * sigmoid(gate[s * E + e]);
            }
        }
        for m in mixed.iter_mut() {
            *m /= HC as f32;
        }
        let inject = w.inject.as_ref().map(|iw| iw.matvec(&xn)).unwrap_or_default();
        (mixed, inject)
    }

    fn combine(r: &mut [f32], y: &[f32], inject: &[f32]) {
        for s in 0..HC {
            let w = 2.0 * sigmoid(inject[s] / HC as f32);
            for e in 0..E {
                r[s * E + e] += y[e] * w;
            }
        }
    }

    fn rope(&self, x: &mut [f32], pos: usize) {
        let n = self.cfg.n_rot;
        for i in 0..n / 2 {
            let theta = pos as f32 * self.cfg.rope_base.powf(-2.0 * i as f32 / n as f32);
            let (s, c) = theta.sin_cos();
            let (x0, x1) = (x[i], x[i + n / 2]);
            x[i] = x0 * c - x1 * s;
            x[i + n / 2] = x0 * s + x1 * c;
        }
    }

    fn gdn(&self, w: &Gdn, x: &[f32], l: usize, st: &mut State, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let c = &self.cfg;
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_dim);
        let ch = self.conv_channels();
        let qkv = w.qkv.matvec(x);
        let z = w.gate.matvec(x);
        let beta: Vec<f32> = w.beta.matvec(x).into_iter().map(sigmoid).collect();
        let g: Vec<f32> = w.alpha.matvec(x).iter().enumerate().map(|(h, a)| softplus(a + w.dt_bias[h]) * w.a[h]).collect();
        // Causal conv over [state(3) | qkv], then silu.
        let win = &mut st.conv[l];
        let mut cv = vec![0f32; ch];
        for i in 0..ch {
            let mut s = w.conv[4 * i + 3] * qkv[i];
            for k in 0..3 {
                s += w.conv[4 * i + k] * win[k * ch + i];
            }
            cv[i] = silu(s);
        }
        win.copy_within(ch.., 0);
        win[2 * ch..].copy_from_slice(&qkv);
        let l2 = |v: &mut [f32]| {
            let ss: f32 = v.iter().map(|a| a * a).sum();
            let r = 1.0 / (ss + EPS).sqrt();
            v.iter_mut().for_each(|a| *a *= r);
        };
        let (q, rest) = cv.split_at_mut(kh * d);
        let (k, v) = rest.split_at_mut(kh * d);
        for h in 0..kh {
            l2(&mut q[h * d..(h + 1) * d]);
            l2(&mut k[h * d..(h + 1) * d]);
        }
        let scale = 1.0 / (d as f32).sqrt();
        let mut o = vec![0f32; vh * d];
        let ssm = &mut st.ssm[l];
        for h in 0..vh {
            let kk = &k[(h % kh) * d..(h % kh + 1) * d];
            let qq = &q[(h % kh) * d..(h % kh + 1) * d];
            let vv = &v[h * d..(h + 1) * d];
            let s = &mut ssm[h * d * d..(h + 1) * d * d];
            let decay = g[h].exp();
            s.iter_mut().for_each(|a| *a *= decay);
            for j in 0..d {
                let row = &mut s[j * d..(j + 1) * d];
                let delta = (vv[j] - dot(row, kk)) * beta[h];
                for i in 0..d {
                    row[i] += kk[i] * delta;
                }
                o[h * d + j] = dot(row, qq) * scale;
            }
        }
        let mut on = vec![0f32; vh * d];
        for h in 0..vh {
            rms(&o[h * d..(h + 1) * d], &w.norm, &mut on[h * d..(h + 1) * d]);
            for j in 0..d {
                on[h * d + j] *= sigmoid(z[h * d + j]);
            }
        }
        act(&format!("final_output-{l}"), &on);
        w.out.matvec(&on)
    }

    fn attn(&self, w: &Attn, x: &[f32], l: usize, st: &mut State, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_head, c.n_head_kv, c.head_dim);
        let pos = st.pos;
        let qg = w.q.matvec(x);
        act(&format!("Qcur_full-{l}"), &qg);
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        for h in 0..nh {
            rms(&qg[h * 2 * hd..h * 2 * hd + hd], &w.q_norm, &mut q[h * hd..(h + 1) * hd]);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[h * 2 * hd + hd..(h + 1) * 2 * hd]);
        }
        act(&format!("Qcur_normed-{l}"), &q);
        for h in 0..nh {
            self.rope(&mut q[h * hd..(h + 1) * hd], pos);
        }
        act(&format!("Qcur-{l}"), &q);
        let kraw = w.k.matvec(x);
        let v = w.v.matvec(x);
        act(&format!("Vcur-{l}"), &v);
        act(&format!("Vcur-{l}"), &v);
        act(&format!("Kcur-{l}"), &kraw);
        let mut k = vec![0f32; nkv * hd];
        for h in 0..nkv {
            rms(&kraw[h * hd..(h + 1) * hd], &w.k_norm, &mut k[h * hd..(h + 1) * hd]);
        }
        act(&format!("Kcur_normed-{l}"), &k);
        for h in 0..nkv {
            self.rope(&mut k[h * hd..(h + 1) * hd], pos);
        }
        act(&format!("Kcur-{l}"), &k);
        st.kc[l].extend_from_slice(&k);
        st.vc[l].extend_from_slice(&v);

        // Indexer: raw key per cell; a block's pooled key once its 4 cells exist.
        let id = c.idx_dim;
        let kr = w.ik.matvec(x);
        st.kr[l].extend_from_slice(&kr);
        let r = pos; // one sequence starting at 0: rank == position
        if (r + 1) % c.kpool == 0 {
            let b = r / c.kpool;
            let mut mean = vec![0f32; id];
            for cell in b * c.kpool..(b + 1) * c.kpool {
                for i in 0..id {
                    mean[i] += st.kr[l][cell * id + i];
                }
            }
            mean.iter_mut().for_each(|a| *a /= c.kpool as f32);
            let mut pk = vec![0f32; id];
            rms(&mean, &w.ik_norm, &mut pk);
            self.rope(&mut pk, b * c.kpool);
            st.kp[l].extend_from_slice(&pk);
        }
        let nv = (r + 1) / c.kpool;
        let n_top = c.idx_top_k / c.kpool;
        let mut cells: Vec<usize> = Vec::new();
        let blocks: Vec<usize> = if nv <= n_top {
            (0..nv).collect()
        } else {
            let iqv = w.iq.matvec(x);
            let mut qi = vec![0f32; c.idx_heads * id];
            for h in 0..c.idx_heads {
                rms(&iqv[h * id..(h + 1) * id], &w.iq_norm, &mut qi[h * id..(h + 1) * id]);
                self.rope(&mut qi[h * id..(h + 1) * id], pos);
            }
            let inv = 1.0 / (id as f32).sqrt();
            let mut sc: Vec<(f32, usize)> = (0..nv)
                .map(|b| {
                    let kb = &st.kp[l][b * id..(b + 1) * id];
                    let s: f32 = (0..c.idx_heads).map(|h| dot(&qi[h * id..(h + 1) * id], kb).max(0.0)).sum();
                    (s * inv, b)
                })
                .collect();
            act(&format!("indexer_score-{l}"), &sc.iter().map(|x| x.0).collect::<Vec<_>>());
            sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            sc.truncate(n_top);
            sc.into_iter().map(|x| x.1).collect()
        };
        for b in blocks {
            cells.extend(b * c.kpool..(b + 1) * c.kpool);
        }
        let n_tail = (r + 1) % c.kpool;
        for t in 0..n_tail {
            cells.push(r - t);
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let mut o = vec![0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / (nh / nkv);
            let qh = &q[h * hd..(h + 1) * hd];
            let sc: Vec<f32> = cells.iter().map(|&cl| dot(qh, &st.kc[l][(cl * nkv + kvh) * hd..(cl * nkv + kvh + 1) * hd]) * scale).collect();
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = p.iter().sum();
            let oh = &mut o[h * hd..(h + 1) * hd];
            for (pi, &cl) in p.iter().zip(&cells) {
                let vr = &st.vc[l][(cl * nkv + kvh) * hd..(cl * nkv + kvh + 1) * hd];
                for i in 0..hd {
                    oh[i] += pi / z * vr[i];
                }
            }
        }
        act(&format!("kqv_out-{l}"), &o);
        act(&format!("attn_pregate-{l}"), &o);
        act(&format!("gate_reshaped-{l}"), &gate);
        for i in 0..nh * hd {
            o[i] *= sigmoid(gate[i]);
        }
        w.o.matvec(&o)
    }

    fn moe(&self, w: &Moe, x: &[f32], l: usize, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let c = &self.cfg;
        let logits = w.router.matvec(x);
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let p: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = p.iter().sum();
        let mut idx: Vec<usize> = (0..c.n_expert).collect();
        idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(c.n_expert_used);
        let wsum: f32 = idx.iter().map(|&e| p[e] / z).sum::<f32>().max(6.103515625e-5);
        let ff = c.ff_exp;
        let mut y = vec![0f32; E];
        for &e in &idx {
            let mut g = vec![0f32; ff];
            let mut u = vec![0f32; ff];
            w.gate.matvec_rows_expert(e * ff, x, &mut g);
            w.up.matvec_rows_expert(e * ff, x, &mut u);
            let [sg, su, sd] = match &w.scale {
                Some([a, b, c]) => [a[e], b[e], c[e]],
                None => [1.0; 3],
            };
            let hmid: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a * sg) * (b * su)).collect();
            let mut d = vec![0f32; E];
            w.down.matvec_rows_expert_down(e * E, &hmid, &mut d);
            if sd != 1.0 {
                d.iter_mut().for_each(|v| *v *= sd);
            }
            let we = p[e] / z / wsum;
            for i in 0..E {
                y[i] += we * d[i];
            }
        }
        act(&format!("ffn_moe_out-{l}"), &y);
        let g = w.sh_gate.matvec(x);
        let u = w.sh_up.matvec(x);
        let hmid: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        let d = w.sh_down.matvec(&hmid);
        let sg = sigmoid(dot(&w.sh_router, x));
        for i in 0..E {
            y[i] += sg * d[i];
        }
        y
    }

    /// The 16 PLE table rows for the token at `st.pos` (already pushed to `st.tokens`).
    pub fn ple_rows(&self, st: &State) -> Vec<u64> {
        ple_rows(&self.cfg, &st.tokens)
    }

    fn ple(&self, w: &Ple, r: &mut [f32], st: &mut State, act: &mut dyn FnMut(&str, &[f32])) {
        let c = &self.cfg;
        let pd = c.ple_dim;
        let rows = self.ple_rows(st);
        let mut emb = vec![0f32; rows.len() * pd];
        for (h, &row) in rows.iter().enumerate() {
            w.table.row(row as usize, &mut emb[h * pd..(h + 1) * pd]);
        }
        act("ple_embd", &emb);
        let key = w.key.matvec(&emb);
        let value = w.value.matvec(&emb);
        let mut gates = [0f32; HC];
        let mut gated = vec![0f32; HCD];
        for s in 0..HC {
            let sl = s * E..(s + 1) * E;
            let mut kn = vec![0f32; E];
            let mut qn = vec![0f32; E];
            rms(&key[sl.clone()], &w.norm_key[sl.clone()], &mut kn);
            rms(&r[sl.clone()], &w.norm_query[sl.clone()], &mut qn);
            let sc = dot(&kn, &qn) / (E as f32).sqrt();
            let sgn = if sc > 0.0 { 1.0 } else if sc < 0.0 { -1.0 } else { 0.0 };
            gates[s] = sigmoid(sgn * sc.abs().max(1e-6).sqrt());
            for e in 0..E {
                gated[s * E + e] = value[e] * gates[s];
            }
        }
        act(&format!("ple_gate-{}", c.ple_layer), &gates);
        let mut cn = vec![0f32; HCD];
        for s in 0..HC {
            let sl = s * E..(s + 1) * E;
            rms(&gated[sl.clone()], &w.norm_conv[sl.clone()], &mut cn[sl]);
        }
        // Depthwise causal conv, kernel 4, dilation 3; history holds the previous 9 inputs, oldest first.
        let dil = c.ple_ngram;
        let kk = c.ple_conv_k;
        let hist_len = (kk - 1) * dil;
        let mut conv = vec![0f32; HCD];
        for ch in 0..HCD {
            let mut s = 0f32;
            for k in 0..kk {
                let back = (kk - 1 - k) * dil;
                let xv = if back == 0 { cn[ch] } else { st.ple_hist[(hist_len - back) * HCD + ch] };
                s += w.conv[k + kk * ch] * xv;
            }
            conv[ch] = silu(s);
        }
        st.ple_hist.copy_within(HCD.., 0);
        st.ple_hist[(hist_len - 1) * HCD..].copy_from_slice(&cn);
        act(&format!("ple_conv_out-{}", c.ple_layer), &conv);
        for i in 0..HCD {
            r[i] += gated[i] + conv[i];
        }
    }

    /// Runs one token at position `st.pos`; returns the logits. `act` receives named activations.
    pub fn forward(&self, tok: u32, st: &mut State, act: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        st.tokens.push(tok);
        let mut e = vec![0f32; E];
        self.embd.row(tok as usize, &mut e);
        act("model.input_embed", &e);
        let mut r = vec![0f32; HCD];
        for s in 0..HC {
            r[s * E..(s + 1) * E].copy_from_slice(&e);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(p) = &layer.ple {
                self.ple(p, &mut r, st, act);
            }
            let (x, inj) = self.mix(&layer.hc_attn, &r);
            act(&format!("hc_mixed-{l}"), &x);
            let y = match &layer.mixer {
                Mixer::Gdn(g) => self.gdn(g, &x, l, st, act),
                Mixer::Attn(a) => self.attn(a, &x, l, st, act),
            };
            act(&format!("{}-{l}", if self.cfg.is_attn(l) { "attn_output" } else { "linear_attn_out" }), &y);
            Self::combine(&mut r, &y, &inj);
            let (x, inj) = self.mix(&layer.hc_ffn, &r);
            act(&format!("hc_mixed-{l}"), &x);
            let y = self.moe(&layer.moe, &x, l, act);
            act(&format!("ffn_out-{l}"), &y);
            Self::combine(&mut r, &y, &inj);
            act(&format!("l_last-{l}"), &r);
        }
        let (x, _) = self.mix(&self.head, &r);
        act("result_norm", &x);
        st.pos += 1;
        self.output.matvec(&x)
    }
}
