//! ggml tensor encodings: block geometry for the types this model and its drafter use.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ2_S,
    IQ3_S,
    IQ4_NL,
    IQ2_XXS,
    IQ3_XXS,
    IQ1_M,
    I32,
    /// NVFP4: 64 E2M1 values in 32 bytes with four UE4M3 sub-block scales (ggml's block_nvfp4; ModelOpt's
    /// per-tensor `weight_scale_2` travels as a separate F32 `.scale` tensor).
    NVFP4,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            16 => Self::IQ2_XXS,
            18 => Self::IQ3_XXS,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            26 => Self::I32,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            40 => Self::NVFP4,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q8_0 => 8,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::IQ2_XXS => 16,
            Self::IQ3_XXS => 18,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::I32 => 26,
            Self::IQ1_M => 29,
            Self::BF16 => 30,
            Self::NVFP4 => 40,
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_S => (256, 82),
            Self::IQ3_XXS => (256, 98),
            Self::IQ3_S => (256, 110),
            Self::IQ4_NL => (32, 18),
            Self::IQ1_M => (256, 56),
            Self::NVFP4 => (64, 36),
        }
    }

    /// Bytes of `n` elements; `n` must be a whole number of blocks.
    pub fn bytes(self, n: usize) -> usize {
        let (be, bb) = self.block();
        assert!(n % be == 0, "{self:?}: {n} elements is not a whole number of {be}-element blocks");
        n / be * bb
    }
}
