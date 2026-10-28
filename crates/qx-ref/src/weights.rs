//! A weight tensor read straight from the memory-mapped GGUF, dequantized row by row.
use qx_pack::ggml::GgmlType;
use qx_pack::gguf::{Gguf, TensorInfo};
use rayon::prelude::*;
use std::sync::OnceLock;

/// `QX_REF_COMPAT=1`: round each matvec's input the way ggml's CPU backend quantizes activations, so the
/// reference can be compared tightly with llama.cpp's outputs. `QX_REF_COMPAT=experts`: round only the
/// routed experts' inputs, as QuixiInfer's CPU expert pool does. Off, every product is exact f32.
pub fn compat() -> bool {
    compat_mode() == 1
}

pub fn compat_mode() -> u8 {
    static C: OnceLock<u8> = OnceLock::new();
    *C.get_or_init(|| match std::env::var("QX_REF_COMPAT").as_deref() {
        Ok("1") => 1,
        Ok("experts") => 2,
        _ => 0,
    })
}

/// ggml's Q8_0 activation rounding: per 32, `d = amax / 127` stored as f16, `q = round(x / d)`.
pub fn round_q8_0(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (xs, os) in x.chunks(32).zip(out.chunks_mut(32)) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let dh = half::f16::from_f32(d).to_f32();
        for (o, v) in os.iter_mut().zip(xs) {
            *o = dh * (v * id).round();
        }
    }
    out
}

/// ggml's Q8_K activation rounding: per 256, scale from the signed value of largest magnitude.
pub fn round_q8_k(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (xs, os) in x.chunks(256).zip(out.chunks_mut(256)) {
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xs {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            continue;
        }
        let iscale = -127.0 / max;
        let d = 1.0 / iscale;
        for (o, v) in os.iter_mut().zip(xs) {
            let q = ((iscale * v).round_ties_even() as i32).min(127);
            *o = d * q as f32;
        }
    }
    out
}

fn compat_input(ty: GgmlType, x: &[f32]) -> Option<Vec<f32>> {
    if !compat() {
        return None;
    }
    match ty {
        GgmlType::Q8_0 | GgmlType::IQ4_NL | GgmlType::NVFP4 => Some(round_q8_0(x)),
        GgmlType::Q6_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ2_S | GgmlType::IQ3_S | GgmlType::IQ2_XXS | GgmlType::IQ3_XXS | GgmlType::IQ1_M => Some(round_q8_k(x)),
        GgmlType::BF16 => Some(x.iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect()),
        GgmlType::F16 => Some(x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect()),
        GgmlType::F32 | GgmlType::I32 => None,
    }
}

#[derive(Clone, Copy)]
pub struct W<'a> {
    pub ty: GgmlType,
    /// Elements per row (ggml ne[0]).
    pub cols: usize,
    /// Rows (product of the other dimensions).
    pub rows: usize,
    pub data: &'a [u8],
}

impl<'a> W<'a> {
    pub fn new(g: &'a Gguf, t: &TensorInfo) -> Self {
        let cols = t.shape[0] as usize;
        let rows = (t.elements() as usize) / cols;
        Self { ty: t.ty, cols, rows, data: g.data(t) }
    }

    pub fn row_bytes(&self) -> usize {
        self.ty.bytes(self.cols)
    }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        qx_pack::dequant(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for r in 0..self.rows {
            self.row(r, &mut v[r * self.cols..(r + 1) * self.cols]);
        }
        v
    }

    /// `y[i] = W[row0 + i] . x` for `i < y.len()`, in f32, rows in parallel.
    pub fn matvec_rows(&self, row0: usize, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        let xq = compat_input(self.ty, x);
        let x = xq.as_deref().unwrap_or(x);
        assert!(row0 + y.len() <= self.rows);
        let cols = self.cols;
        y.par_chunks_mut(64).enumerate().for_each_init(
            || vec![0f32; cols],
            |buf, (ci, ys)| {
                for (k, yv) in ys.iter_mut().enumerate() {
                    self.row(row0 + ci * 64 + k, buf);
                    *yv = dot(buf, x);
                }
            },
        );
    }

    /// A routed expert's matvec: its input is rounded in either compat mode.
    pub fn matvec_rows_expert(&self, row0: usize, x: &[f32], y: &mut [f32]) {
        if compat_mode() == 2 {
            let xq = match self.ty {
                GgmlType::IQ4_NL | GgmlType::Q8_0 => round_q8_0(x),
                _ => round_q8_k(x),
            };
            self.matvec_rows(row0, &xq, y)
        } else {
            self.matvec_rows(row0, x, y)
        }
    }

    /// A routed expert's down projection: QuixiInfer's pool feeds it Q8_0-rounded activations whatever the format.
    pub fn matvec_rows_expert_down(&self, row0: usize, x: &[f32], y: &mut [f32]) {
        if compat_mode() == 2 {
            self.matvec_rows(row0, &round_q8_0(x), y)
        } else {
            self.matvec_rows(row0, x, y)
        }
    }

    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; self.rows];
        self.matvec_rows(0, x, &mut y);
        y
    }
}

/// f32 dot product with eight partial sums, so the compiler vectorises it.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for k in 0..8 {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}
