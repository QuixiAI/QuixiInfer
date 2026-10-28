#!/usr/bin/env python3
"""Concurrent load against an OpenAI-compatible server: `loadgen.py [--url URL] [--levels 1,4,8,16] [--max-tokens N]`.

At each concurrency level, that many clients each send requests back to back (streamed chat completions, varied
prompts) until every client has finished `--rounds` requests. `--warmup N` runs N unmeasured requests first. Reports aggregate output tok/s, mean per-request
decode tok/s, and time to first token (median, p90).
"""
import argparse, json, statistics, threading, time, urllib.request

PROMPTS = [
    "Explain how a hash map handles collisions, with an example.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main causes of the French Revolution? Give a structured answer.",
    "Write a Python function that merges two sorted lists, and explain its complexity.",
    "Describe the water cycle to a ten-year-old.",
    "Compare TCP and UDP and say when you would use each.",
    "Summarize the plot of Hamlet in a few paragraphs.",
    "Give me a week-long beginner running plan with explanations.",
    "How does public-key cryptography work? Use an analogy.",
    "List ten creative names for a coffee shop and explain each one.",
    "What is the difference between a virus and a bacterium?",
    "Write a haiku sequence about autumn in the mountains.",
    "Explain gradient descent and why the learning rate matters.",
    "Draft a polite email asking a landlord to fix a leaking tap.",
    "What happened during the Apollo 13 mission?",
    "Explain recursion with three different examples.",
]

API_KEY = None
FOLLOWUPS = ["Can you elaborate on the second point?", "Give a concrete example.", "Summarize that in two sentences.", "What are the common mistakes here?", "Now explain it for an expert audience."]

def one(url, prompt, max_tokens, temperature, unique=False, history=None, think=True):
    if unique:
        import uuid
        prompt = f"Request {uuid.uuid4().hex}: {prompt}"
    sent_prompt = prompt
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    req = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    if not think:
        req["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(req).encode()
    headers = {"content-type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(url + "/chat/completions", body, headers)
    t0 = time.time()
    first, n, usage, text = None, 0, None, []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            ev = json.loads(d)
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content"):
                    text.append(delta["content"])
                if delta.get("content") or delta.get("reasoning_content"):
                    n += 1
                    if first is None:
                        first = time.time()
    t1 = time.time()
    toks = usage["completion_tokens"] if usage else n
    return dict(ttft=(first or t1) - t0, total=t1 - t0, tokens=toks, decode=(t1 - (first or t1)), prompt_tokens=usage["prompt_tokens"] if usage else 0, text="".join(text), sent_prompt=sent_prompt)

def level(url, c, rounds, max_tokens, temperature, unique=False, turns=1):
    res, lock = [], threading.Lock()
    def client(i):
        for k in range(rounds):
            # a conversation of `turns` turns: each follow-up prompt carries the whole history (no thinking, so the
            # history the template renders is what the model generated)
            history = []
            for t in range(turns):
                q = PROMPTS[(i * rounds + k) % len(PROMPTS)] if t == 0 else FOLLOWUPS[(t - 1) % len(FOLLOWUPS)]
                r = one(url, q, max_tokens, temperature, unique and t == 0, history, think=turns == 1)
                history = history + [{"role": "user", "content": (r.get("sent_prompt") or q)}, {"role": "assistant", "content": r["text"]}]
                with lock:
                    res.append(r)
    t0 = time.time()
    th = [threading.Thread(target=client, args=(i,)) for i in range(c)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    toks = sum(r["tokens"] for r in res)
    ttft = sorted(r["ttft"] for r in res)
    per = [(r["tokens"] - 1) / r["decode"] for r in res if r["decode"] > 0 and r["tokens"] > 1]
    return dict(c=c, requests=len(res), tokens=toks, wall=wall, agg=toks / wall, per_req=statistics.mean(per) if per else 0,
                ttft_med=statistics.median(ttft), ttft_p90=ttft[min(len(ttft) - 1, int(0.9 * len(ttft)))], prompt_tokens=sum(r["prompt_tokens"] for r in res))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=0, help="unmeasured requests first, 16 at a time (lets the expert cache adapt)")
    ap.add_argument("--unique", action="store_true", help="prefix every prompt with a nonce, so no prompt repeats (no prefix-cache hits)")
    ap.add_argument("--turns", type=int, default=1, help="turns per conversation (follow-ups carry the history; thinking off)")
    ap.add_argument("--api-key", default=None, help="bearer token")
    a = ap.parse_args()
    global API_KEY
    API_KEY = a.api_key
    if a.warmup:
        level(a.url, min(16, a.warmup), max(1, a.warmup // 16), a.max_tokens, a.temperature, a.unique, a.turns)
    print(f"{'conc':>4} {'reqs':>4} {'tokens':>6} {'prompt':>7} {'wall s':>7} {'agg tok/s':>9} {'per-req tok/s':>13} {'TTFT med':>8} {'TTFT p90':>8}", flush=True)
    for c in [int(x) for x in a.levels.split(",")]:
        r = level(a.url, c, a.rounds, a.max_tokens, a.temperature, a.unique, a.turns)
        print(f"{r['c']:>4} {r['requests']:>4} {r['tokens']:>6} {r['prompt_tokens']:>7} {r['wall']:>7.1f} {r['agg']:>9.1f} {r['per_req']:>13.1f} {r['ttft_med']:>8.2f} {r['ttft_p90']:>8.2f}", flush=True)

if __name__ == "__main__":
    main()
