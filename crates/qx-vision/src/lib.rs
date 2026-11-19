//! The vision tower of DeepSeek-V4.1-Flash (projector type `deepseek41v`), on the CPU in f32.
//!
//! An image becomes a block of LLM input embeddings, one row of `dim` floats per position:
//!
//! ```text
//! [IMAGE_START] ([IMAGE] * n_llm_w [IMAGE_NEW_LINE]) * n_llm_h [IMAGE_END]
//! ```
//!
//! Written from the model's own reference (`inference/vision.py`, `inference/image_processor.py` in
//! deepseek-ai/DeepSeek-V4.1-Flash-1017): a native-resolution ViT (RMSNorm, SwiGLU, 2D rotary positions, no class
//! token), then the aligner (3x3 patch merge as `F.unfold`, two linear layers with an erf GELU between). The
//! weights come from the `mmproj` GGUF. The image is fitted as PIL's `ImageOps.pad` does it (Pillow's bicubic
//! resampling in fixed point, grey 127 bars), which `preprocess` reproduces byte for byte.
use anyhow::{anyhow, bail, Result};
use qx_pack::gguf::{Gguf, TensorInfo};
use rayon::prelude::*;

unsafe extern "C" {
    fn erf(x: f64) -> f64;
}

struct Block {
    ln1: Vec<f32>,
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    ln2: Vec<f32>,
    gate_w: Vec<f32>,
    up_w: Vec<f32>,
    down_w: Vec<f32>,
}

pub struct Vision {
    pub dim: usize,
    pub out_dim: usize,
    heads: usize,
    ffn: usize,
    pub patch: usize,
    pub merge: usize,
    eps: f32,
    theta: f32,
    pub min_pixels: usize,
    pub max_tokens: usize,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    blocks: Vec<Block>,
    post_ln: Vec<f32>,
    mm1_w: Vec<f32>,
    mm1_b: Vec<f32>,
    mm2_w: Vec<f32>,
    mm2_b: Vec<f32>,
    start: Vec<f32>,
    end: Vec<f32>,
    newline: Vec<f32>,
}

/// A fitted image: `h` x `w` pixels (multiples of the patch size), RGB bytes.
pub struct Fitted {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

fn tensor_f32(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<u64>)> {
    let t: &TensorInfo = g.tensor(name).ok_or_else(|| anyhow!("the projector file has no tensor {name}"))?;
    let raw = g.data(t);
    let n = t.elements() as usize;
    let v: Vec<f32> = if raw.len() == n * 4 {
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    } else if raw.len() == n * 2 {
        raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()
    } else {
        bail!("tensor {name}: {} bytes for {n} elements (neither f32 nor f16)", raw.len());
    };
    Ok((v, t.shape.clone()))
}

/// y (n x out) = x (n x inp) . w^T + b, `w` stored as `out` rows of `inp`; rows split over the thread pool.
fn linear(x: &[f32], n: usize, inp: usize, w: &[f32], out: usize, b: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), n * inp);
    assert_eq!(w.len(), out * inp);
    let mut y = vec![0f32; n * out];
    let per = n.div_ceil(rayon::current_num_threads().max(1)).max(8);
    y.par_chunks_mut(per * out).zip(x.par_chunks(per * inp)).for_each(|(yc, xc)| {
        let rows = xc.len() / inp;
        // w^T as a strided view: element (k, j) of the right operand is w[j * inp + k]
        unsafe { matrixmultiply::sgemm(rows, inp, out, 1.0, xc.as_ptr(), inp as isize, 1, w.as_ptr(), 1, inp as isize, 0.0, yc.as_mut_ptr(), out as isize, 1) };
        if let Some(b) = b {
            for r in yc.chunks_exact_mut(out) {
                for (v, bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        }
    });
    y
}

fn rms_norm(x: &[f32], d: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    y.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(yr, xr)| {
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let s = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            yr[i] = w[i] * (xr[i] * s);
        }
    });
    y
}

impl Vision {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let s = |k: &str| g.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
        if s("clip.projector_type").as_deref() != Some("deepseek41v") {
            bail!("{}: projector type {:?}, not deepseek41v", path.display(), s("clip.projector_type"));
        }
        let u = |k: &str| g.u64(k).map(|v| v as usize);
        let (dim, heads, ffn, layers) = (u("clip.vision.embedding_length")?, u("clip.vision.attention.head_count")?, u("clip.vision.feed_forward_length")?, u("clip.vision.block_count")?);
        let (patch, merge, out_dim) = (u("clip.vision.patch_size")?, u("clip.vision.projector.scale_factor")?, u("clip.vision.projection_dim")?);
        let min_pixels = u("clip.vision.image_min_pixels")?;
        let eps = g.get("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        let f = |n: &str| tensor_f32(&g, n).map(|t| t.0);
        let mut blocks = Vec::with_capacity(layers);
        for l in 0..layers {
            let b = |n: &str| f(&format!("v.blk.{l}.{n}"));
            blocks.push(Block { ln1: b("ln1.weight")?, qkv_w: b("attn_qkv.weight")?, qkv_b: b("attn_qkv.bias")?, out_w: b("attn_out.weight")?, out_b: b("attn_out.bias")?, ln2: b("ln2.weight")?, gate_w: b("ffn_gate.weight")?, up_w: b("ffn_up.weight")?, down_w: b("ffn_down.weight")? });
        }
        let me = Self {
            dim, out_dim, heads, ffn, patch, merge, eps,
            // (the reference's vision_rope_theta; the GGUF does not carry it and llama.cpp sets the same constant)
            theta: 10000.0,
            min_pixels,
            // vision_max_n_token of the model's config: the whole block, delimiters and newlines included
            max_tokens: 1024,
            patch_w: f("v.patch_embd.weight")?, patch_b: f("v.patch_embd.bias")?, blocks, post_ln: f("v.post_ln.weight")?,
            mm1_w: f("mm.1.weight")?, mm1_b: f("mm.1.bias")?, mm2_w: f("mm.2.weight")?, mm2_b: f("mm.2.bias")?,
            start: f("v.token_embd.img_start")?, end: f("v.token_embd.img_end")?, newline: f("v.image_newline")?,
        };
        if me.patch_w.len() != dim * 3 * patch * patch || me.mm1_w.len() != out_dim * dim * merge * merge || me.blocks[0].qkv_w.len() != 3 * dim * dim || me.blocks[0].gate_w.len() != ffn * dim || dim % heads != 0 || (dim / heads) % 4 != 0 {
            bail!("{}: tensor shapes do not fit the header", path.display());
        }
        Ok(me)
    }

    /// The LLM grid (rows, columns) of a fitted image of `h` x `w` pixels.
    pub fn llm_grid(&self, h: usize, w: usize) -> (usize, usize) {
        ((h / self.patch).div_ceil(self.merge), (w / self.patch).div_ceil(self.merge))
    }

    /// Positions an image of that grid takes in the prompt.
    pub fn block_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
        n_llm_h * (n_llm_w + 1) + 2
    }

    /// `plan_image_grid` of the reference: the pixel size (height, width) an image of `width` x `height` is
    /// fitted to.
    pub fn plan(&self, width: usize, height: usize) -> (usize, usize) {
        let (p, r) = (self.patch, self.merge);
        let (mut w, mut h) = (width, height);
        if w * h > 0 && w * h < self.min_pixels {
            let ratio = (self.min_pixels as f64 / (w * h) as f64).sqrt();
            w = (w as f64 * ratio) as usize;
            h = (h as f64 * ratio) as usize;
        }
        let (mut bw, mut bh) = (w.div_ceil(p) * p, h.div_ceil(p) * p);
        let tokens = |bh: usize, bw: usize| { let (gh, gw) = self.llm_grid(bh, bw); Self::block_tokens(gh, gw) };
        if tokens(bh, bw) > self.max_tokens {
            // solve_resize_ratio: the largest size of the same aspect whose block fits
            let ratio = h as f64 / w as f64;
            let max_w = ((self.max_tokens - 2) as f64 / ratio + 0.25).sqrt() - 0.5;
            let max_h = max_w * ratio;
            let cell = p * r;
            if max_w < 1.0 {
                bh = (self.max_tokens - 2) / 2 * cell;
                bw = cell;
            } else if max_h < 1.0 {
                bh = cell;
                bw = (self.max_tokens - 3) * cell;
            } else {
                let beta = (max_w.floor() * cell as f64 / w as f64).min(max_h.floor() * cell as f64 / h as f64);
                bh = (h as f64 * beta / p as f64).floor() as usize * p;
                bw = (w as f64 * beta / p as f64).floor() as usize * p;
            }
            assert!(tokens(bh, bw) <= self.max_tokens);
        }
        (bh, bw)
    }

    /// Decodes an image file (PNG, JPEG, WebP, GIF, BMP) and fits it: PIL's `convert("RGB")`, then
    /// `ImageOps.pad(image, (w, h), color=(127, 127, 127))`.
    pub fn preprocess(&self, bytes: &[u8]) -> Result<Fitted> {
        let img = image::load_from_memory(bytes).map_err(|e| anyhow!("cannot decode the image: {e}"))?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w == 0 || h == 0 {
            bail!("empty image");
        }
        self.fit(img.as_raw(), w, h)
    }

    /// `preprocess` from RGB bytes.
    pub fn fit(&self, rgb: &[u8], w: usize, h: usize) -> Result<Fitted> {
        let (bh, bw) = self.plan(w, h);
        if bh == 0 || bw == 0 {
            bail!("an image of {w}x{h} pixels is too thin to fit");
        }
        // ImageOps.pad: the image resized to fit inside, centred on the colour
        let round_even = |x: f64| -> usize { x.round_ties_even() as usize };
        let (mut nw, mut nh) = (bw, bh);
        let (im_ratio, dest_ratio) = (w as f64 / h as f64, bw as f64 / bh as f64);
        if im_ratio != dest_ratio {
            if im_ratio > dest_ratio {
                nh = round_even(h as f64 / w as f64 * bw as f64);
            } else {
                nw = round_even(w as f64 / h as f64 * bh as f64);
            }
        }
        if nw == 0 || nh == 0 {
            bail!("an image of {w}x{h} pixels is too thin to fit");
        }
        let resized = if (nw, nh) == (w, h) { rgb.to_vec() } else { resize_bicubic(rgb, w, h, nw, nh) };
        if (nw, nh) == (bw, bh) {
            return Ok(Fitted { w: bw, h: bh, rgb: resized });
        }
        let mut out = vec![127u8; bw * bh * 3];
        let (ox, oy) = (round_even((bw - nw) as f64 * 0.5), round_even((bh - nh) as f64 * 0.5));
        for y in 0..nh {
            out[((y + oy) * bw + ox) * 3..((y + oy) * bw + ox + nw) * 3].copy_from_slice(&resized[y * nw * 3..(y + 1) * nw * 3]);
        }
        Ok(Fitted { w: bw, h: bh, rgb: out })
    }

    /// The block of a fitted image: (rows of `out_dim` floats, LLM grid rows, columns).
    pub fn encode(&self, im: &Fitted) -> (Vec<f32>, usize, usize) {
        let (p, d) = (self.patch, self.dim);
        let (nh, nw) = (im.h / p, im.w / p);
        let n = nh * nw;
        // patches, each flattened (channel, y, x); pixels as the reference feeds them: (v / 255 - 0.5) / 0.5 rounded to bf16
        let px = |v: u8| -> f32 {
            let x = (v as f32 / 255.0 - 0.5) / 0.5;
            let b = x.to_bits();
            f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
        };
        let pin = 3 * p * p;
        let mut patches = vec![0f32; n * pin];
        patches.par_chunks_mut(pin).enumerate().for_each(|(i, dst)| {
            let (gy, gx) = (i / nw, i % nw);
            for c in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        dst[(c * p + y) * p + x] = px(im.rgb[((gy * p + y) * im.w + gx * p + x) * 3 + c]);
                    }
                }
            }
        });
        let mut x = linear(&patches, n, pin, &self.patch_w, d, Some(&self.patch_b));
        // 2D rotary angles per patch: head_dim / 4 frequencies for the row, then as many for the column
        let hd = d / self.heads;
        let nf = hd / 4;
        let inv: Vec<f32> = (0..nf).map(|j| 1.0 / self.theta.powf(2.0 * j as f32 / (2 * nf) as f32)).collect();
        let mut cs = vec![(0f32, 0f32); n * 2 * nf];
        for i in 0..n {
            let (r, c) = ((i / nw) as f32, (i % nw) as f32);
            for j in 0..nf {
                let (a, b) = (r * inv[j], c * inv[j]);
                cs[i * 2 * nf + j] = (a.cos(), a.sin());
                cs[i * 2 * nf + nf + j] = (b.cos(), b.sin());
            }
        }
        for b in &self.blocks {
            let h = rms_norm(&x, d, &b.ln1, self.eps);
            let a = self.attention(b, &h, n, &cs);
            x.par_iter_mut().zip(a.par_iter()).for_each(|(v, a)| *v += a);
            let h = rms_norm(&x, d, &b.ln2, self.eps);
            let mut gate = linear(&h, n, d, &b.gate_w, self.ffn, None);
            let up = linear(&h, n, d, &b.up_w, self.ffn, None);
            gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, u)| *g = *g / (1.0 + (-*g).exp()) * u);
            let m = linear(&gate, n, self.ffn, &b.down_w, d, None);
            x.par_iter_mut().zip(m.par_iter()).for_each(|(v, a)| *v += a);
        }
        let x = rms_norm(&x, d, &self.post_ln, self.eps);
        // the aligner: the grid zero-padded to a multiple of the merge, each merge x merge cell flattened as
        // F.unfold does (feature-major: feature c of cell patch (ky, kx) at c * merge^2 + ky * merge + kx)
        let r = self.merge;
        let (gh, gw) = (nh.div_ceil(r), nw.div_ceil(r));
        let cin = d * r * r;
        let mut cells = vec![0f32; gh * gw * cin];
        cells.par_chunks_mut(cin).enumerate().for_each(|(i, dst)| {
            let (gy, gx) = (i / gw, i % gw);
            for ky in 0..r {
                for kx in 0..r {
                    let (y, xx) = (gy * r + ky, gx * r + kx);
                    if y < nh && xx < nw {
                        let src = &x[(y * nw + xx) * d..(y * nw + xx + 1) * d];
                        for c in 0..d {
                            dst[c * r * r + ky * r + kx] = src[c];
                        }
                    }
                }
            }
        });
        let mut h1 = linear(&cells, gh * gw, cin, &self.mm1_w, self.out_dim, Some(&self.mm1_b));
        h1.par_iter_mut().for_each(|v| *v = (0.5 * *v as f64 * (1.0 + unsafe { erf(*v as f64 / std::f64::consts::SQRT_2) })) as f32);
        let feat = linear(&h1, gh * gw, self.out_dim, &self.mm2_w, self.out_dim, Some(&self.mm2_b));
        let e = self.out_dim;
        let mut out = Vec::with_capacity(Self::block_tokens(gh, gw) * e);
        out.extend_from_slice(&self.start);
        for y in 0..gh {
            out.extend_from_slice(&feat[y * gw * e..(y + 1) * gw * e]);
            out.extend_from_slice(&self.newline);
        }
        out.extend_from_slice(&self.end);
        (out, gh, gw)
    }

    /// Self-attention over all `n` patches: q and k rotated by the 2D positions, softmax(q k^T / sqrt(head_dim)) v.
    fn attention(&self, b: &Block, h: &[f32], n: usize, cs: &[(f32, f32)]) -> Vec<f32> {
        let d = self.dim;
        let hd = d / self.heads;
        let half = hd / 2;
        let qkv = linear(h, n, d, &b.qkv_w, 3 * d, Some(&b.qkv_b));
        // per head: q and k rotated (n x hd), v (n x hd)
        let heads: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..self.heads)
            .into_par_iter()
            .map(|hh| {
                let (mut q, mut k, mut v) = (vec![0f32; n * hd], vec![0f32; n * hd], vec![0f32; n * hd]);
                for i in 0..n {
                    let row = &qkv[i * 3 * d..(i + 1) * 3 * d];
                    let (qs, ks, vs) = (&row[hh * hd..(hh + 1) * hd], &row[d + hh * hd..d + (hh + 1) * hd], &row[2 * d + hh * hd..2 * d + (hh + 1) * hd]);
                    for j in 0..half {
                        let (c, s) = cs[i * half + j];
                        q[i * hd + j] = qs[j] * c - qs[half + j] * s;
                        q[i * hd + half + j] = qs[half + j] * c + qs[j] * s;
                        k[i * hd + j] = ks[j] * c - ks[half + j] * s;
                        k[i * hd + half + j] = ks[half + j] * c + ks[j] * s;
                    }
                    v[i * hd..(i + 1) * hd].copy_from_slice(vs);
                }
                (q, k, v)
            })
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        // (head, block of query rows) tasks: scores (rows x n), softmax, times v
        const QB: usize = 256;
        let mut o = vec![0f32; n * d];
        o.par_chunks_mut(QB * d).enumerate().for_each(|(bi, oc)| {
            let r0 = bi * QB;
            let rows = oc.len() / d;
            let mut sc = vec![0f32; rows * n];
            let mut ov = vec![0f32; rows * hd];
            for (hh, (q, k, v)) in heads.iter().enumerate() {
                unsafe { matrixmultiply::sgemm(rows, hd, n, scale, q.as_ptr().add(r0 * hd), hd as isize, 1, k.as_ptr(), 1, hd as isize, 0.0, sc.as_mut_ptr(), n as isize, 1) };
                for r in sc.chunks_exact_mut(n) {
                    let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for x in r.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    let inv = 1.0 / sum;
                    for x in r.iter_mut() {
                        *x *= inv;
                    }
                }
                unsafe { matrixmultiply::sgemm(rows, n, hd, 1.0, sc.as_ptr(), n as isize, 1, v.as_ptr(), hd as isize, 1, 0.0, ov.as_mut_ptr(), hd as isize, 1) };
                for r in 0..rows {
                    oc[r * d + hh * hd..r * d + (hh + 1) * hd].copy_from_slice(&ov[r * hd..(r + 1) * hd]);
                }
            }
        });
        linear(&o, n, d, &b.out_w, d, Some(&b.out_b))
    }
}

/// Pillow's bicubic resampling of an RGB image (libImaging/Resample.c: separable, horizontal then vertical, 22-bit
/// fixed-point coefficients, a = -0.5, the filter widened when shrinking), byte for byte.
pub fn resize_bicubic(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    const BITS: u32 = 32 - 8 - 2;
    fn filter(x: f64) -> f64 {
        let x = x.abs();
        let a = -0.5;
        if x < 1.0 { ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 } else if x < 2.0 { (((x - 5.0) * x + 8.0) * x - 4.0) * a } else { 0.0 }
    }
    // per output pixel: the first input pixel, their number, the coefficients
    fn coeffs(inp: usize, out: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
        let scale = inp as f64 / out as f64;
        let fscale = scale.max(1.0);
        let support = 2.0 * fscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = Vec::with_capacity(out);
        let mut kk = vec![0i32; out * ksize];
        let mut pre = vec![0f64; ksize];
        for xx in 0..out {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / fscale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(inp as i64) as usize - xmin;
            let mut ww = 0.0;
            for x in 0..xmax {
                let wv = filter((x as f64 + xmin as f64 - center + 0.5) * ss);
                pre[x] = wv;
                ww += wv;
            }
            for x in 0..xmax {
                let v = if ww != 0.0 { pre[x] / ww } else { pre[x] };
                kk[xx * ksize + x] = (v * (1u64 << BITS) as f64 + if v < 0.0 { -0.5 } else { 0.5 }) as i32;
            }
            bounds.push((xmin, xmax));
        }
        (ksize, bounds, kk)
    }
    let clip = |v: i32| -> u8 { (v >> BITS).clamp(0, 255) as u8 };
    // Pillow runs a pass only when that dimension changes
    let hor: Vec<u8> = if nw != w {
        let (ks, bounds, kk) = coeffs(w, nw);
        let mut out = vec![0u8; nw * h * 3];
        out.par_chunks_mut(nw * 3).enumerate().for_each(|(y, row)| {
            let s = &src[y * w * 3..(y + 1) * w * 3];
            for xx in 0..nw {
                let (xmin, cnt) = bounds[xx];
                let mut acc = [1i32 << (BITS - 1); 3];
                for x in 0..cnt {
                    let k = kk[xx * ks + x];
                    for c in 0..3 {
                        acc[c] += s[(xmin + x) * 3 + c] as i32 * k;
                    }
                }
                for c in 0..3 {
                    row[xx * 3 + c] = clip(acc[c]);
                }
            }
        });
        out
    } else {
        src.to_vec()
    };
    if nh == h {
        return hor;
    }
    let (ks, bounds, kk) = coeffs(h, nh);
    let mut out = vec![0u8; nw * nh * 3];
    out.par_chunks_mut(nw * 3).enumerate().for_each(|(yy, row)| {
        let (ymin, cnt) = bounds[yy];
        for i in 0..nw * 3 {
            let mut acc = 1i32 << (BITS - 1);
            for y in 0..cnt {
                acc += hor[(ymin + y) * nw * 3 + i] as i32 * kk[yy * ks + y];
            }
            row[i] = clip(acc);
        }
    });
    out
}
