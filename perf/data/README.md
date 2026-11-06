# Benchmark inputs

Little-endian u32 token ids for the DeepSeek-V4.1-Flash tokenizer, vendored here so the harness runs on any box
(`golden/data/` is git-ignored).

| file | tokens | source |
|---|--:|---|
| `route1k.tokens` | 1,024 | the first 1,024 tokens of the routing trace used for every recorded DeepSeek number since 2026-11-04 (`golden/data/route1k.tokens`) |
| `route16k.tokens` | 16,384 | the 16k continuation of the same trace (`golden/data/route16k.tokens`) |

Do not regenerate or edit them: every scoreboard row depends on them being the same bytes.
| `wiki256k.tokens` | 262,400 | wikitext-2 `wiki.test.raw` + `wiki.valid.raw`, tokenized with llama.cpp's `llama-tokenize` (`--no-bos`), the first 262,400 tokens; for the 256k context checks (2026-11-06) |
