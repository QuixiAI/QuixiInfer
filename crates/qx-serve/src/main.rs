//! QuixiInfer: `quixiinfer serve` (OpenAI-compatible HTTP API on 127.0.0.1) and `quixiinfer run` (one prompt,
//! streamed to the terminal).
mod chat;
mod worker;
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

const MODEL_ID: &str = "qwen3.8-flash-next";

struct App {
    jobs: Mutex<mpsc::Sender<Job>>,
    tok: &'static qx_tok::Tok,
    tpl: chat::Template,
    api_key: Option<String>,
    /// Tokens of context per sequence: the default completion limit is what the prompt leaves of it.
    max_ctx: usize,
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
    Json(json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "quixiinfer"}]})).into_response()
}

async fn generate(app: Arc<App>, prompt: Vec<u32>, body: &Value, chat: bool, thinking: bool) -> Response {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    // no limit given: the rest of the context (a thinking model spends thousands of tokens before its answer;
    // a 4096 default cut long answers off inside the reasoning)
    let max_tokens = body.get("max_completion_tokens").or(body.get("max_tokens")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or_else(|| app.max_ctx.saturating_sub(prompt.len() + 1));
    let grammar = match grammar::Spec::from_body(body) {
        Ok(g) => g,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let job = Job { arrived: std::time::Instant::now(), prompt, grammar, max_tokens, sampling: sampling_from(body), stop: stops_from(body), events: tx };
    if app.jobs.lock().unwrap().send(job).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    let id = format!("{}-{:x}", if chat { "chatcmpl" } else { "cmpl" }, now() ^ (std::process::id() as u64) << 20);
    let created = now();
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut split = ThinkSplit { in_think: chat && thinking, pending: String::new(), started: false };
    if stream {
        let s = async_stream_events(rx, id, created, chat, split);
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
                    json!({"id": id, "object": "chat.completion", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
                } else {
                    json!({"id": id, "object": "text_completion", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "text": content, "finish_reason": finish}], "usage": usage})
                };
                return Json(out).into_response();
            }
        }
    }
    err(StatusCode::INTERNAL_SERVER_ERROR, "engine ended the request")
}

fn async_stream_events(rx: tokio::sync::mpsc::UnboundedReceiver<Event>, id: String, created: u64, chat: bool, mut split: ThinkSplit) -> impl futures::Stream<Item = Result<SseEvent, std::convert::Infallible>> {
    let mut first = true;
    tokio_stream::wrappers::UnboundedReceiverStream::new(rx).flat_map(move |ev| {
        let mut out: Vec<Result<SseEvent, std::convert::Infallible>> = Vec::new();
        let chunk = |delta: Value, finish: Value, usage: Option<Value>| {
            let mut v = if chat {
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            } else {
                json!({"id": id, "object": "text_completion", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish}]})
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
    let prompt = match app.tok.encode(&text) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    if std::env::var("QX_SCHED_STATS").is_ok() {
        eprintln!("chat: template {:.0} ms, tokenizer {:.0} ms ({} chars, {} tokens)", 1e3 * (t1 - t0).as_secs_f64(), 1e3 * t1.elapsed().as_secs_f64(), text.len(), prompt.len());
    }
    generate(app.clone(), prompt, &body, true, thinking).await
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
    generate(app.clone(), prompt, &body, false, false).await
}

fn usage() -> ! {
    eprintln!("usage:\n  quixiinfer serve [--host 127.0.0.1] [--port 8080] [--ctx 32768] [--slots 16] [--prefill-budget-ms 1000] [--profile RANKING] [--api-key KEY] [--model GGUF]\n  quixiinfer run --prompt TEXT [--max-tokens N] [--temperature T] [--no-think] [--raw] [--ctx N] [--profile RANKING]");
    std::process::exit(2)
}

fn main() -> Result<()> {
    qx_gpu::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("serve") => {
            let host = arg(&args, "--host").unwrap_or_else(|| "127.0.0.1".into());
            let api_key = arg(&args, "--api-key");
            if host != "127.0.0.1" && host != "localhost" && api_key.is_none() {
                return Err(anyhow!("refusing to listen on {host} without --api-key"));
            }
            let port: u16 = arg(&args, "--port").map(|s| s.parse()).transpose()?.unwrap_or(8080);
            let (jobs, tok, tpl, ctx) = load(&args)?;
            let app = Arc::new(App { jobs: Mutex::new(jobs), tok, tpl, api_key, max_ctx: ctx });
            let router = Router::new()
                .route("/v1/models", get(models))
                .route("/v1/chat/completions", post(chat_completions))
                .route("/v1/completions", post(completions))
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
            jobs.send(Job { arrived: std::time::Instant::now(), prompt, grammar: None, max_tokens, sampling, stop: Vec::new(), events: tx })?;
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
