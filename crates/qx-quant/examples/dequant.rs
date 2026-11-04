//! Dequantizes raw GGUF blocks from a file: `dequant TYPE IN.bin OUT.f32`. Used by tools/dequant_check.py
//! to compare every decoder against gguf-py on real tensors.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (ty, src) = (&a[1], std::fs::read(&a[2]).expect("read"));
    let (bytes, elems, f): (usize, usize, fn(&[u8], &mut [f32])) = match ty.as_str() {
        "Q8_0" => (34, 32, qx_quant::dequant_q8_0),
        "Q6_K" => (210, 256, qx_quant::dequant_q6_k),
        "Q5_K" => (176, 256, qx_quant::dequant_q5_k),
        "Q2_K" => (84, 256, qx_quant::dequant_q2_k),
        "IQ4_NL" => (18, 32, qx_quant::dequant_iq4_nl),
        "IQ3_S" => (110, 256, qx_quant::dequant_iq3_s),
        "IQ3_XXS" => (98, 256, qx_quant::dequant_iq3_xxs),
        "IQ2_S" => (82, 256, qx_quant::dequant_iq2_s),
        "IQ2_XS" => (74, 256, qx_quant::dequant_iq2_xs),
        "IQ2_XXS" => (66, 256, qx_quant::dequant_iq2_xxs),
        "IQ1_S" => (50, 256, qx_quant::dequant_iq1_s),
        _ => panic!("unknown type {ty}"),
    };
    let n = src.len() / bytes;
    let mut out = vec![0f32; n * elems];
    f(&src[..n * bytes], &mut out);
    let raw: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[3], raw).expect("write");
}
