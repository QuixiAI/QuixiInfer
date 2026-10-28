//! The model's tokenizer, built from the GGUF: GPT-2 byte-level BPE with Qwen 3.5's pre-tokenizer split, and the
//! control tokens as special tokens.
use anyhow::{anyhow, Result};
use qx_pack::gguf::Gguf;
use std::collections::HashMap;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// Qwen 3.5's split regex, as in its tokenizer.json.
const QWEN35_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tok {
    pub inner: Tokenizer,
    pub eos: u32,
    pub specials: HashMap<String, u32>,
}

impl Tok {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arr = |k: &str| g.get(k).and_then(|v| v.as_arr()).ok_or_else(|| anyhow!("missing {k}"));
        let tokens: Vec<String> = arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let types: Vec<u64> = arr("tokenizer.ggml.token_type")?.iter().map(|v| v.as_u64().unwrap_or(1)).collect();
        let merges: Vec<(String, String)> = arr("tokenizer.ggml.merges")?
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(|m| m.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
            .collect();
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let bpe = BPE::builder().vocab_and_merges(vocab.into_iter().collect::<ahash::AHashMap<String, u32>>(), merges).build().map_err(|e| anyhow!("BPE: {e}"))?;
        let mut tk = Tokenizer::new(bpe);
        let split = Split::new(SplitPattern::Regex(QWEN35_SPLIT.into()), SplitDelimiterBehavior::Isolated, false).map_err(|e| anyhow!("split: {e}"))?;
        tk.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ]))));
        tk.with_decoder(Some(ByteLevelDecoder::default()));
        // ggml token types: 3 control, 4 user-defined; both match as whole tokens before BPE.
        let mut specials = HashMap::new();
        let added: Vec<AddedToken> = tokens
            .iter()
            .zip(&types)
            .enumerate()
            .filter(|(_, (_, t))| **t == 3 || **t == 4)
            .map(|(i, (s, &t))| {
                specials.insert(s.clone(), i as u32);
                AddedToken::from(s.clone(), t == 3)
            })
            .collect();
        tk.add_special_tokens(&added);
        let eos = g.u64("tokenizer.ggml.eos_token_id")? as u32;
        Ok(Self { inner: tk, eos, specials })
    }

    /// Token ids of `text`; special-token text (e.g. `<|im_start|>`) becomes its token.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.inner.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }
}
