//! Reading GGUF models and QuixiInfer's pack format.
pub mod ggml;
pub mod gguf;
pub mod pack;

use ggml::GgmlType;

/// Decodes `src` (whole blocks of `ty`) into `out`.
pub fn dequant(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    use qx_quant::*;
    match ty {
        GgmlType::F32 => dequant_f32(src, out),
        GgmlType::F16 => dequant_f16(src, out),
        GgmlType::BF16 => dequant_bf16(src, out),
        GgmlType::Q8_0 => dequant_q8_0(src, out),
        GgmlType::Q6_K => dequant_q6_k(src, out),
        GgmlType::IQ4_NL => dequant_iq4_nl(src, out),
        GgmlType::IQ3_S => dequant_iq3_s(src, out),
        GgmlType::IQ2_S => dequant_iq2_s(src, out),
        GgmlType::NVFP4 => dequant_nvfp4(src, out),
        t => panic!("no dequantizer for {t:?}"),
    }
}
