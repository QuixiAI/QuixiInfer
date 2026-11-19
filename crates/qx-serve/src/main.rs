//! QuixiInfer: `quixiinfer serve` (OpenAI-compatible HTTP API on 127.0.0.1) and `quixiinfer run` (one prompt,
//! streamed to the terminal).
mod chat;
mod worker;
mod worker_ds;
mod grammar;

use anyhow::{anyhow, Result};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use qx_engine::engine::Sampling;
use serde_json::{json, Value};
use std::sync::{mpsc, Arc, Mutex};
use worker::{Event, Job};


struct App {
    jobs: Mutex<mpsc::Sender<Job>>,
    tok: &'static qx_tok::Tok,
    tpl: chat::Template,
    api_key: Option<String>,
    /// Tokens of context per sequence: the default completion limit is what the prompt leaves of it.
    max_ctx: usize,
    /// The model id the API reports: the GGUF's `general.name`.
    model_id: String,
    /// The vision tower (`--mmproj`), the id of the chat template's image placeholder token, the blocks of the
    /// last few images (by the hash of their bytes), and the lock that lets one image be encoded at a time.
    vision: Option<Arc<qx_vision::Vision>>,
    image_tok: Option<u32>,
    vis_cache: Mutex<Vec<(u64, Arc<Vec<f32>>)>>,
    vis_lock: tokio::sync::Mutex<()>,
    /// The vision tower's own threads: free to run on every CPU the process may use (the engine pins its threads,
    /// and a thread started from a pinned one inherits its one CPU: the tower then took 14 s where it takes 2).
    vis_pool: Option<Arc<rayon::ThreadPool>>,
}

/// The chat template's image placeholder (one token).
const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// Standard base64 (padding and whitespace tolerated).
fn base64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return Err("invalid base64 in an image".into()),
        };
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// The bytes of one image part: an OpenAI `image_url` (a `data:` URL) or an Anthropic `image` (`source.data`).
fn image_bytes(part: &Value) -> std::result::Result<Vec<u8>, String> {
    let url = part.pointer("/image_url/url").or_else(|| part.get("image_url")).or_else(|| part.pointer("/source/url")).or_else(|| part.get("url")).and_then(|v| v.as_str());
    if let Some(d) = part.pointer("/source/data").or_else(|| part.get("data")).and_then(|v| v.as_str()) {
        return base64_decode(d);
    }
    let Some(url) = url else { return Err("an image part has neither image_url.url nor source.data".into()) };
    if let Some(rest) = url.strip_prefix("data:") {
        let Some((header, payload)) = rest.split_once(',') else { return Err("malformed data: URL in an image part".into()) };
        if !header.contains(";base64") {
            return Err("an image data: URL must be base64".into());
        }
        return base64_decode(payload);
    }
    Err("this server does not fetch image URLs: send the image inline as a data: URL (data:image/png;base64,...)".into())
}

/// The image parts of the messages in the order the chat template writes their placeholders: message by message,
/// part by part, the parts of a tool result in place.
fn collect_images(messages: &Value, out: &mut Vec<Vec<u8>>) -> std::result::Result<(), String> {
    fn parts(content: &Value, out: &mut Vec<Vec<u8>>) -> std::result::Result<(), String> {
        let Some(arr) = content.as_array() else { return Ok(()) };
        for p in arr {
            match p.get("type").and_then(|v| v.as_str()) {
                Some("image") | Some("image_url") => out.push(image_bytes(p)?),
                Some("tool_result") => {
                    if let Some(c) = p.get("content") {
                        parts(c, out)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    for m in messages.as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if let Some(c) = m.get("content") {
            parts(c, out)?;
        }
    }
    Ok(())
}

/// Images above this many bytes, or more of them in one request, are refused.
const IMAGE_MAX_BYTES: usize = 20 << 20;
const IMAGES_MAX: usize = 16;

/// The cached block of the image with this key; a hit becomes the most recent entry.
fn cached_block(app: &App, key: u64) -> Option<Arc<Vec<f32>>> {
    let mut c = app.vis_cache.lock().unwrap();
    let i = c.iter().position(|(k, _)| *k == key)?;
    let hit = c.remove(i);
    let rows = hit.1.clone();
    c.push(hit);
    Some(rows)
}

/// The block of one image (rows of the model's embedding width), from the cache or the vision tower. The encode
/// runs in a task of its own, one at a time, and its result goes to the cache whether or not the request that
/// asked is still there (a client that gives up and asks again finds it instead of starting another encode).
async fn image_block(app: &Arc<App>, bytes: Vec<u8>) -> std::result::Result<(u64, Arc<Vec<f32>>), String> {
    use std::hash::{Hash, Hasher};
    let vision = app.vision.clone().ok_or("this server was started without the vision projector (--mmproj): it cannot take images")?;
    if bytes.len() > IMAGE_MAX_BYTES {
        return Err(format!("an image of {} bytes is over the limit of {IMAGE_MAX_BYTES}", bytes.len()));
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    let key = h.finish();
    if let Some(r) = cached_block(app, key) {
        return Ok((key, r));
    }
    let app = app.clone();
    tokio::spawn(async move {
        let _one = app.vis_lock.lock().await;
        if let Some(r) = cached_block(&app, key) {
            return Ok((key, r));
        }
        let t0 = std::time::Instant::now();
        let pool = app.vis_pool.clone().expect("vision pool");
        let (rows, gh, gw, w, h) = tokio::task::spawn_blocking(move || -> std::result::Result<_, String> {
            pool.install(|| {
                let im = vision.preprocess(&bytes).map_err(|e| format!("{e:#}"))?;
                let (rows, gh, gw) = vision.encode(&im);
                Ok((rows, gh, gw, im.w, im.h))
            })
        })
        .await
        .map_err(|e| format!("vision tower: {e}"))??;
        if std::env::var("QX_SCHED_STATS").is_ok() {
            eprintln!("vision: an image fitted to {w}x{h}, grid {gh}x{gw}, {} positions in {:.2} s", qx_vision::Vision::block_tokens(gh, gw), t0.elapsed().as_secs_f64());
        }
        let rows = Arc::new(rows);
        let mut c = app.vis_cache.lock().unwrap();
        c.push((key, rows.clone()));
        // the most recent blocks, VIS_CACHE_FLOATS of them at most (and always the newest)
        while c.len() > 1 && c.iter().map(|(_, r)| r.len()).sum::<usize>() > VIS_CACHE_FLOATS {
            c.remove(0);
        }
        Ok((key, rows))
    })
    .await
    .map_err(|e| format!("vision task: {e}"))?
}

/// Floats of image blocks kept for images sent again (335 MB: about 16 images of the largest size, 80 typical).
const VIS_CACHE_FLOATS: usize = 16384 * 5120;

/// Requests without the key end here, before their body is read.
async fn auth_layer(State(app): State<Arc<App>>, req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    if !authorized(&app, req.headers()) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    next.run(req).await
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
}

fn model_path(args: &[String]) -> String {
    arg(args, "--model").unwrap_or_else(qx_engine::default_model_path)
}

/// Loads the model, tokenizer, chat template and the engine thread. The expert cache takes the VRAM the rest leaves.
fn load(args: &[String]) -> Result<(mpsc::Sender<Job>, &'static qx_tok::Tok, chat::Template, usize)> {
    let g: &'static qx_pack::gguf::Gguf = Box::leak(Box::new(qx_pack::gguf::Gguf::open(std::path::Path::new(&model_path(args)))?));
    let tok: &'static qx_tok::Tok = Box::leak(Box::new(qx_tok::Tok::from_gguf(g)?));
    let tpl = chat::Template::new(g.get("tokenizer.chat_template").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("the GGUF has no chat template"))?.to_string())?;
    let ctx: usize = arg(args, "--ctx").map(|s| s.parse()).transpose()?.unwrap_or(32768);
    if g.get("general.architecture").and_then(|v| v.as_str()) == Some("deepseek41") {
        // DeepSeek-V4.1-Flash: the three-tier engine on the pack (`--pack` or QX_DS_PACK), one request at a time
        let gib = |k: &str, flag: &str, d: f64| -> Result<f64> { Ok(match arg(args, flag) { Some(v) => v.parse()?, None => std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }) };
        let pack = arg(args, "--pack").or_else(|| std::env::var("QX_DS_PACK").ok()).ok_or_else(|| anyhow!("DeepSeek needs the pack: --pack FILE.qxp or QX_DS_PACK"))?;
        let profile = arg(args, "--profile").or_else(|| std::env::var("QX_DS_PROFILE").ok()).or_else(|| std::path::Path::new("golden/profile-dsv41.txt").exists().then(|| "golden/profile-dsv41.txt".to_string()));
        // 128 since 2026-11-06 (was 64): with the full staged share and a 256-record staging area a 128-token chunk
        // prefills 10.8 ms per token against 12.3 at 64, a 16k prompt 18.2 against 21.8; the staging area costs the
        // VRAM expert tier about 1.5 GiB, about 10% of per-request decode at three concurrent requests
        // (perf/optimization_status.md, "128-chunk bundle"). `--window 64` with QX_DS_STAGE_SHARE=6 QX_DS_STAGE_CAP=96
        // is the previous configuration.
        let window: usize = arg(args, "--window").map(|s| s.parse()).transpose()?.unwrap_or(128);
        let profile_out = arg(args, "--profile-out").or_else(|| std::env::var("QX_DS_PROFILE_OUT").ok());
        // a ranking saved by an earlier run is what the tiers should start from
        let profile = match &profile_out {
            Some(po) if std::path::Path::new(po).exists() => Some(po.clone()),
            _ => profile,
        };
        let slots: usize = arg(args, "--slots").map(|s| s.parse()).transpose()?.unwrap_or(2);
        // the chunk path for long prompts (worker_ds: prompt-only steps go through forward_chunk): 4096 rows by
        // default on servers with 32k of context or more. Since the elastic flex tail (2026-11-08) its buffers borrow
        // the VRAM arena's tail for a prompt's duration instead of taking it from the expert tier for good, so the
        // chunk size no longer costs decode: measured at ctx 262144 with 3 slots, time to first token on a
        // 13.5k-token prompt 42.3 s at 4096 against 35.5 at 8192, decode under 3 streams 107.1 / 107.2 ms per token
        // (was 109.5 / 112.6 with fixed buffers), both at the full 12.2 GiB tier. 8192 is the default; 0 or empty: off
        if std::env::var_os("QX_DS_CHUNK").is_none() && ctx >= 32768 {
            // SAFETY: before the engine thread starts (single-threaded here); the engine reads it at load
            unsafe { std::env::set_var("QX_DS_CHUNK", "8192") };
        }
        let chunk = std::env::var("QX_DS_CHUNK").ok().filter(|v| !v.is_empty() && v != "0");
        eprintln!("quixiinfer: loading DeepSeek-V4.1-Flash (context {ctx}, {slots} slots, prompt windows of {window}{}, tiers ranked by {})", chunk.as_deref().map(|c| format!(", prompt chunks of {c}")).unwrap_or_default(), profile.as_deref().unwrap_or("layer order"));
        let opt = worker_ds::Options { max_ctx: ctx, window, slots, pack: pack.into(), vram_gib: gib("QX_DS_VRAM", "--vram-gib", 21.0)?, ram_gib: gib("QX_DS_RAM", "--ram-gib", 100.0)?, profile: profile.map(Into::into), profile_out: profile_out.map(Into::into) };
        let jobs = worker_ds::start(g, tok, opt)?;
        return Ok((jobs, tok, tpl, ctx));
    }
    let order: Vec<(usize, usize)> = match arg(args, "--profile") {
        Some(p) => std::fs::read_to_string(p)?
            .lines()
            .map(|l| {
                let v: Vec<usize> = l.split_whitespace().take(2).map(|x| x.parse().unwrap()).collect();
                (v[0], v[1])
            })
            .collect(),
        None => qx_engine::routing::default_order(g, 48, 512),
    };
    // the engine sizes the expert cache from the VRAM left once everything else is allocated
    let budget = qx_gpu::mem::mem_info().0;
    eprintln!("quixiinfer: loading (context {ctx} per sequence)");
    let slots: usize = arg(args, "--slots").map(|s| s.parse()).transpose()?.unwrap_or(16);
    let prefill_budget_ms: f64 = arg(args, "--prefill-budget-ms").map(|s| s.parse()).transpose()?.unwrap_or(1000.0);
    let jobs = worker::start(g, tok, ctx, budget, order, worker::Options { slots, prefill_budget_ms })?;
    Ok((jobs, tok, tpl, ctx))
}

fn sampling_from(body: &Value) -> Sampling {
    let f = |k: &str, d: f64| body.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    Sampling {
        temperature: f("temperature", 1.0) as f32,
        top_p: f("top_p", 0.95) as f32,
        min_p: f("min_p", 0.0) as f32,
        top_k: f("top_k", 20.0) as u32,
        seed: body.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| now() ^ 0x5eed),
    }
}

fn stops_from(body: &Value) -> Vec<String> {
    match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    }
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

fn authorized(app: &App, h: &HeaderMap) -> bool {
    match &app.api_key {
        None => true,
        Some(k) => h.get("authorization").and_then(|v| v.to_str().ok()) == Some(&format!("Bearer {k}")),
    }
}

/// Splits streamed text into reasoning (before `</think>`) and content; blank lines opening the content are dropped.
struct ThinkSplit {
    in_think: bool,
    pending: String,
    started: bool,
}

impl ThinkSplit {
    fn push(&mut self, delta: &str) -> (String, String) {
        let (r, c) = self.split(delta);
        if self.started || c.is_empty() {
            return (r, c);
        }
        let c = c.trim_start_matches(['\n', '\r']).to_string();
        self.started = !c.is_empty();
        (r, c)
    }

    fn split(&mut self, delta: &str) -> (String, String) {
        if !self.in_think {
            return (String::new(), delta.to_string());
        }
        self.pending.push_str(delta);
        if let Some(i) = self.pending.find("</think>") {
            let r = self.pending[..i].to_string();
            let c = self.pending[i + "</think>".len()..].trim_start_matches('\n').to_string();
            self.in_think = false;
            self.pending.clear();
            return (r, c);
        }
        // keep back a possible partial "</think>"
        let keep = (1..="</think>".len()).rev().find(|&n| self.pending.ends_with(&"</think>"[..n])).unwrap_or(0);
        let out = self.pending[..self.pending.len() - keep].to_string();
        self.pending = self.pending[self.pending.len() - keep..].to_string();
        (out, String::new())
    }
}

async fn models(State(app): State<Arc<App>>, h: HeaderMap) -> Response {
    if !authorized(&app, &h) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    Json(json!({"object": "list", "data": [{"id": app.model_id, "object": "model", "created": 0, "owned_by": "quixiinfer"}]})).into_response()
}

async fn generate(app: Arc<App>, prompt: Vec<u32>, images: Vec<worker::JobImage>, body: &Value, chat: bool, thinking: bool) -> Response {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    // no limit given: the rest of the context (a thinking model spends thousands of tokens before its answer;
    // a 4096 default cut long answers off inside the reasoning)
    let max_tokens = body.get("max_completion_tokens").or(body.get("max_tokens")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or_else(|| app.max_ctx.saturating_sub(prompt.len() + 1));
    let grammar = match grammar::Spec::from_body(body) {
        Ok(g) => g,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let job = Job { arrived: std::time::Instant::now(), prompt, grammar, grammar_after_think: chat && thinking, max_tokens, sampling: sampling_from(body), stop: stops_from(body), events: tx, images };
    if app.jobs.lock().unwrap().send(job).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    let id = format!("{}-{:x}", if chat { "chatcmpl" } else { "cmpl" }, now() ^ (std::process::id() as u64) << 20);
    let created = now();
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut split = ThinkSplit { in_think: chat && thinking, pending: String::new(), started: false };
    if stream {
        let s = async_stream_events(rx, id, created, chat, split, app.model_id.clone());
        return Sse::new(s).into_response();
    }
    let (mut reasoning, mut content) = (String::new(), String::new());
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Delta(d) => {
                let (r, c) = split.push(&d);
                reasoning.push_str(&r);
                content.push_str(&c);
            }
            Event::Error(e) => return err(StatusCode::BAD_REQUEST, e),
            Event::Done { prompt_tokens, completion_tokens, finish, prompt_tps, decode_tps } => {
                let usage = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens,
                                   "prompt_tokens_per_second": prompt_tps, "completion_tokens_per_second": decode_tps});
                if !split.pending.is_empty() {
                    reasoning.push_str(&split.pending);
                }
                let out = if chat {
                    let mut msg = json!({"role": "assistant", "content": content});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = json!(reasoning);
                    }
                    json!({"id": id, "object": "chat.completion", "created": created, "model": app.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
                } else {
                    json!({"id": id, "object": "text_completion", "created": created, "model": app.model_id, "choices": [{"index": 0, "text": content, "finish_reason": finish}], "usage": usage})
                };
                return Json(out).into_response();
            }
        }
    }
    err(StatusCode::INTERNAL_SERVER_ERROR, "engine ended the request")
}

fn async_stream_events(rx: tokio::sync::mpsc::UnboundedReceiver<Event>, id: String, created: u64, chat: bool, mut split: ThinkSplit, model: String) -> impl futures::Stream<Item = Result<SseEvent, std::convert::Infallible>> {
    let mut first = true;
    tokio_stream::wrappers::UnboundedReceiverStream::new(rx).flat_map(move |ev| {
        let mut out: Vec<Result<SseEvent, std::convert::Infallible>> = Vec::new();
        let chunk = |delta: Value, finish: Value, usage: Option<Value>| {
            let mut v = if chat {
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            } else {
                json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish}]})
            };
            if let Some(u) = usage {
                v["usage"] = u;
            }
            Ok(SseEvent::default().data(v.to_string()))
        };
        match ev {
            Event::Delta(d) => {
                let (r, c) = split.push(&d);
                let mut delta = json!({});
                if first && chat {
                    delta["role"] = json!("assistant");
                    first = false;
                }
                if !r.is_empty() {
                    delta["reasoning_content"] = json!(r);
                }
                if !c.is_empty() {
                    delta["content"] = json!(c);
                }
                if delta.as_object().is_some_and(|o| !o.is_empty()) {
                    out.push(chunk(delta, Value::Null, None));
                }
            }
            Event::Error(e) => out.push(Ok(SseEvent::default().data(json!({"error": {"message": e}}).to_string()))),
            Event::Done { prompt_tokens, completion_tokens, finish, prompt_tps, decode_tps } => {
                let usage = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens,
                                   "prompt_tokens_per_second": prompt_tps, "completion_tokens_per_second": decode_tps});
                out.push(chunk(json!({}), json!(finish), Some(usage)));
                out.push(Ok(SseEvent::default().data("[DONE]")));
            }
        }
        futures::stream::iter(out)
    })
}

async fn chat_completions(State(app): State<Arc<App>>, h: HeaderMap, Json(body): Json<Value>) -> Response {
    if !authorized(&app, &h) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    let Some(messages) = body.get("messages") else { return err(StatusCode::BAD_REQUEST, "messages is required") };
    let thinking = body.pointer("/chat_template_kwargs/enable_thinking").and_then(|v| v.as_bool()).unwrap_or(true);
    let t0 = std::time::Instant::now();
    let text = match app.tpl.render(messages, body.get("tools"), thinking) {
        Ok(t) => t,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let t1 = std::time::Instant::now();
    let mut prompt = match app.tok.encode(&text) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    // images: each placeholder token becomes the image's block of positions
    let mut image_files = Vec::new();
    if let Err(e) = collect_images(messages, &mut image_files) {
        return err(StatusCode::BAD_REQUEST, e);
    }
    let placeholders = app.image_tok.map_or(0, |it| prompt.iter().filter(|&&t| t == it).count());
    let mut images = Vec::new();
    if !image_files.is_empty() || placeholders > 0 {
        let Some(it) = app.image_tok.filter(|_| app.vision.is_some()) else {
            return err(StatusCode::BAD_REQUEST, "this server was started without the vision projector (--mmproj): it cannot take images");
        };
        if image_files.len() != placeholders {
            return err(StatusCode::BAD_REQUEST, format!("the request has {} images but its rendered prompt has {placeholders} image placeholders (images belong in their own content parts)", image_files.len()));
        }
        if image_files.len() > IMAGES_MAX {
            return err(StatusCode::BAD_REQUEST, format!("{} images in one request (at most {IMAGES_MAX})", image_files.len()));
        }
        let mut blocks = Vec::new();
        for bytes in image_files {
            match image_block(&app, bytes).await {
                Ok(b) => blocks.push(b),
                Err(e) => return err(StatusCode::BAD_REQUEST, e),
            }
        }
        let width = app.vision.as_ref().unwrap().out_dim;
        let mut expanded = Vec::with_capacity(prompt.len() + blocks.iter().map(|b| b.1.len() / width).sum::<usize>());
        let mut next = blocks.into_iter();
        for &t in &prompt {
            if t == it {
                let (key, rows) = next.next().unwrap();
                let n = rows.len() / width;
                images.push(worker::JobImage { at: expanded.len(), n, key, rows });
                expanded.extend(std::iter::repeat_n(it, n));
            } else {
                expanded.push(t);
            }
        }
        prompt = expanded;
    }
    if std::env::var("QX_SCHED_STATS").is_ok() {
        eprintln!("chat: template {:.0} ms, tokenizer {:.0} ms ({} chars, {} tokens)", 1e3 * (t1 - t0).as_secs_f64(), 1e3 * t1.elapsed().as_secs_f64(), text.len(), prompt.len());
    }
    // the reply opens inside a reasoning block only when the rendered prompt ends with one left open (not after an
    // assistant prefill, a task token or a lone system message): the split of the reply and a structured-output
    // constraint's start both follow that, not the request's flag
    let think_open = thinking && text.trim_end().ends_with("<think>");
    generate(app.clone(), prompt, images, &body, true, think_open).await
}

async fn completions(State(app): State<Arc<App>>, h: HeaderMap, Json(body): Json<Value>) -> Response {
    if !authorized(&app, &h) {
        return err(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    let Some(p) = body.get("prompt").and_then(|v| v.as_str()) else { return err(StatusCode::BAD_REQUEST, "prompt must be a string") };
    let prompt = match app.tok.encode(p) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => return err(StatusCode::BAD_REQUEST, "empty prompt"),
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    generate(app.clone(), prompt, Vec::new(), &body, false, false).await
}

const HELP: &str = "\
quixiinfer: an OpenAI-compatible inference server

usage:
  quixiinfer serve [options]            serve /v1/models, /v1/chat/completions and /v1/completions
  quixiinfer run --prompt TEXT [options]  answer one prompt on stdout
  quixiinfer help | --help | -h         this text (loads nothing)

options of both commands:
  --model GGUF          the model file (or QX_MODEL)
  --ctx N               context per sequence in tokens (default 32768)
  --slots N             concurrent sequences (default 16; DeepSeek 2)
  --profile RANKING     expert ranking the tiers start from
                        (DeepSeek: else QX_DS_PROFILE, else golden/profile-dsv41.txt if present)
  --prefill-budget-ms N prompt time allowed per step beside decoders (default 1000; not DeepSeek)

serve:
  --host ADDR           address to listen on (default 127.0.0.1; any other needs --api-key)
  --port N              port (default 8080)
  --api-key KEY         bearer token the clients must send

run:
  --prompt TEXT         the prompt (required)
  --max-tokens N        tokens to generate at most (default 512)
  --temperature T       sampling temperature (default 0: greedy)
  --no-think            render the chat template with reasoning off
  --raw                 feed the prompt as written, without the chat template

DeepSeek-V4.1-Flash models:
  --pack FILE.qxp       the expert pack (required; or QX_DS_PACK)
  --window N            prompt tokens per step (default 128)
  --vram-gib G          VRAM budget in GiB (default 21; or QX_DS_VRAM)
  --ram-gib G           RAM expert tier in GiB (default 100; or QX_DS_RAM)
  --profile-out FILE    save the expert ranking here and start from it next time (or QX_DS_PROFILE_OUT)
  --mmproj FILE         the vision projector GGUF for image input (or QX_DS_MMPROJ; default: mmproj-F16.gguf
                        in or above the model's directory; the word none: no image input)

Further settings are environment variables (QX_DS_*); handoff.md lists the current ones.
";

fn usage() -> ! {
    eprint!("{HELP}");
    std::process::exit(2)
}

/// True when the command line asks for help: `help` as the command, or `--help` / `-h` anywhere except as the text
/// of `--prompt`.
fn wants_help(args: &[String]) -> bool {
    args.get(1).is_some_and(|a| a == "help") || args.iter().enumerate().skip(1).any(|(i, a)| (a == "--help" || a == "-h") && args[i - 1] != "--prompt")
}

fn main() -> Result<()> {
    // the CPUs this process may use, before the engine pins anything
    let mut all_cpus: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut all_cpus) };
    let args: Vec<String> = std::env::args().collect();
    // before anything touches the GPU or the model: `serve --help` used to load the default model
    if wants_help(&args) {
        print!("{HELP}");
        return Ok(());
    }
    if !matches!(args.get(1).map(|s| s.as_str()), Some("serve" | "run")) {
        usage();
    }
    qx_engine::tune_host_allocator();
    qx_gpu::init();
    match args.get(1).map(|s| s.as_str()) {
        Some("serve") => {
            let host = arg(&args, "--host").unwrap_or_else(|| "127.0.0.1".into());
            let api_key = arg(&args, "--api-key");
            if host != "127.0.0.1" && host != "localhost" && api_key.is_none() {
                return Err(anyhow!("refusing to listen on {host} without --api-key"));
            }
            let port: u16 = arg(&args, "--port").map(|s| s.parse()).transpose()?.unwrap_or(8080);
            let (jobs, tok, tpl, ctx) = load(&args)?;
            let model_id = qx_pack::gguf::Gguf::open(std::path::Path::new(&model_path(&args))).ok().and_then(|g| g.get("general.name").and_then(|v| v.as_str()).map(|s| s.to_lowercase().replace(' ', "-"))).unwrap_or_else(|| "quixiinfer".into());
            // the vision tower: --mmproj FILE, QX_DS_MMPROJ, or an mmproj-F16.gguf beside the model's directory
            let model_dir = std::path::Path::new(&model_path(&args)).parent().map(|p| p.to_path_buf()).unwrap_or_default();
            let mmproj = arg(&args, "--mmproj").or_else(|| std::env::var("QX_DS_MMPROJ").ok()).map(std::path::PathBuf::from).or_else(|| [model_dir.join("mmproj-F16.gguf"), model_dir.join("../mmproj-F16.gguf")].into_iter().find(|p| p.exists()));
            let image_tok = tok.encode(IMAGE_PLACEHOLDER).ok().filter(|t| t.len() == 1).map(|t| t[0]);
            let vision = match (&mmproj, image_tok) {
                (Some(p), Some(_)) if p.as_os_str() != "none" => {
                    let v = qx_vision::Vision::load(p)?;
                    // (a projector of another model would hand the engine rows of the wrong width)
                    let width = qx_pack::gguf::Gguf::open(std::path::Path::new(&model_path(&args))).ok().and_then(|g| g.tensor("token_embd.weight").map(|t| t.shape[0] as usize));
                    if width.is_some_and(|w| w != v.out_dim) {
                        return Err(anyhow!("{}: the projector's rows are {} wide, the model's embeddings {}", p.display(), v.out_dim, width.unwrap()));
                    }
                    eprintln!("quixiinfer: vision tower from {} (images of up to {} positions, encoded on the CPU)", p.display(), v.max_tokens);
                    Some(Arc::new(v))
                }
                _ => {
                    eprintln!("quixiinfer: no vision projector (--mmproj): requests with images are refused");
                    None
                }
            };
            let vis_pool = match &vision {
                Some(_) => {
                    let n = std::env::var("QX_VISION_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                    Some(Arc::new(rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("qx-vision-{i}")).start_handler(move |_| unsafe {
                        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &all_cpus);
                    }).build()?))
                }
                None => None,
            };
            let app = Arc::new(App { jobs: Mutex::new(jobs), tok, tpl, api_key, max_ctx: ctx, model_id, vision, image_tok, vis_cache: Mutex::new(Vec::new()), vis_lock: tokio::sync::Mutex::new(()), vis_pool });
            let router = Router::new()
                .route("/v1/models", get(models))
                .route("/v1/chat/completions", post(chat_completions))
                .route("/v1/completions", post(completions))
                // (axum's default body limit is 2 MB: a photograph in base64 is more)
                .layer(axum::extract::DefaultBodyLimit::max(96 << 20))
                .layer(axum::middleware::from_fn_with_state(app.clone(), auth_layer))
                .with_state(app);
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
                eprintln!("quixiinfer: listening on http://{host}:{port}/v1");
                axum::serve(listener, router).await?;
                Ok::<(), anyhow::Error>(())
            })?;
            Ok(())
        }
        Some("run") => {
            let prompt_text = arg(&args, "--prompt").unwrap_or_else(|| usage());
            let max_tokens: usize = arg(&args, "--max-tokens").map(|s| s.parse()).transpose()?.unwrap_or(512);
            let temperature: f32 = arg(&args, "--temperature").map(|s| s.parse()).transpose()?.unwrap_or(0.0);
            let raw = args.iter().any(|a| a == "--raw");
            let thinking = !args.iter().any(|a| a == "--no-think");
            let (jobs, tok, tpl, _) = load(&args)?;
            let text = if raw { prompt_text } else { tpl.render(&json!([{"role": "user", "content": prompt_text}]), None, thinking)? };
            let prompt = tok.encode(&text)?;
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let sampling = Sampling { temperature, top_p: 0.95, min_p: 0.0, top_k: if temperature > 0.0 { 20 } else { 1 }, seed: 1234 };
            jobs.send(Job { arrived: std::time::Instant::now(), prompt, grammar: None, grammar_after_think: false, max_tokens, sampling, stop: Vec::new(), events: tx, images: Vec::new() })?;
            use std::io::Write;
            while let Some(ev) = rx.blocking_recv() {
                match ev {
                    Event::Delta(d) => {
                        print!("{d}");
                        std::io::stdout().flush()?;
                    }
                    Event::Error(e) => return Err(anyhow!(e)),
                    Event::Done { prompt_tokens, completion_tokens, finish, prompt_tps, decode_tps } => {
                        eprintln!("\n[{prompt_tokens} prompt tokens at {prompt_tps:.1} tok/s, {completion_tokens} generated at {decode_tps:.1} tok/s, {finish}]");
                        // leave without running exit handlers: the engine thread may still be saving the request's
                        // prefix, and CUDA's teardown under it made that thread panic
                        std::io::stdout().flush()?;
                        unsafe { libc::_exit(0) };
                    }
                }
            }
            Ok(())
        }
        _ => usage(),
    }
}
