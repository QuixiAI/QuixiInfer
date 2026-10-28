//! Structured outputs: a request's JSON schema, regular expression or Lark grammar compiled (by llguidance) into a
//! token-level matcher that gives the sampler, at every step, the set of tokens that keep the output valid.
use anyhow::{anyhow, Result};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use qx_tok::Tok;
use serde_json::Value;
use std::sync::Arc;

/// What a request constrains its output to.
#[derive(Clone, Debug)]
pub enum Spec {
    JsonSchema(Value),
    Regex(String),
    Lark(String),
}

impl Spec {
    /// The constraint a request body asks for: OpenAI's `response_format` (`json_schema` with its schema, or
    /// `json_object` for any JSON), or vLLM's `guided_json` / `guided_regex` / `guided_grammar`.
    pub fn from_body(body: &Value) -> Result<Option<Self>> {
        if let Some(rf) = body.get("response_format") {
            match rf.get("type").and_then(|t| t.as_str()) {
                Some("json_schema") => {
                    let schema = rf.pointer("/json_schema/schema").or_else(|| rf.get("schema")).cloned().ok_or_else(|| anyhow!("response_format.json_schema.schema is required"))?;
                    return Ok(Some(Spec::JsonSchema(schema)));
                }
                Some("json_object") => return Ok(Some(Spec::JsonSchema(serde_json::json!({"type": "object"})))),
                Some("text") | None => {}
                Some(t) => return Err(anyhow!("unknown response_format type {t}")),
            }
        }
        if let Some(j) = body.get("guided_json") {
            let schema = match j {
                Value::String(s) => serde_json::from_str(s)?,
                v => v.clone(),
            };
            return Ok(Some(Spec::JsonSchema(schema)));
        }
        if let Some(r) = body.get("guided_regex").and_then(|v| v.as_str()) {
            return Ok(Some(Spec::Regex(r.to_string())));
        }
        if let Some(g) = body.get("guided_grammar").and_then(|v| v.as_str()) {
            return Ok(Some(Spec::Lark(g.to_string())));
        }
        Ok(None)
    }
}

/// The compiled tokenizer the matchers work on.
pub struct Grammars {
    factory: ParserFactory,
    pub words: usize,
}

impl Grammars {
    /// Builds the token trie from the tokenizer (its tokens' bytes) with `n_vocab` entries (the model's output
    /// rows; tokens past the tokenizer's are never allowed).
    pub fn new(tok: &Tok, n_vocab: usize) -> Result<Self> {
        let json: Value = serde_json::from_str(&tok.inner.to_string(false).map_err(|e| anyhow!("tokenizer json: {e}"))?)?;
        let mut words = llguidance::token_bytes_from_tokenizer_json(&json)?;
        words.resize(n_vocab, Vec::new());
        let info = TokRxInfo::new(n_vocab as u32, tok.eos);
        let trie = TokTrie::from(&info, &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env)?;
        factory.quiet();
        Ok(Self { factory, words: n_vocab.div_ceil(32) })
    }

    /// A matcher for `spec`, ready to give the first mask.
    pub fn matcher(&self, spec: &Spec) -> Result<Matcher> {
        let g = match spec {
            Spec::JsonSchema(s) => TopLevelGrammar::from_json_schema(s.clone()),
            Spec::Regex(r) => TopLevelGrammar::from_regex(r),
            Spec::Lark(l) => TopLevelGrammar::from_lark(l.clone()),
        };
        let parser = self.factory.create_parser(g);
        let m = Matcher::new(parser);
        if let Some(e) = m.get_error() {
            return Err(anyhow!("grammar: {e}"));
        }
        Ok(m)
    }
}
