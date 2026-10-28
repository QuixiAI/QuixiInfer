#!/usr/bin/env python3
"""One long prompt through the OpenAI-compatible API: `longprompt.py --chars N [--url URL] [--api-key KEY]
[--max-tokens 64]`. Sends the first N characters of a text file as a user message (streamed), reports prompt
tokens (from usage), time to first token, prefill tok/s and decode tok/s after it."""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--api-key", default=None)
ap.add_argument("--file", default="golden/data/wikitext-2-raw/wiki.train.raw")
ap.add_argument("--chars", type=int, default=100000)
ap.add_argument("--max-tokens", type=int, default=64)
a = ap.parse_args()
text = open(a.file, encoding="utf-8").read()[: a.chars]
body = {"model": "default", "messages": [{"role": "user", "content": text + "\n\nSummarize the above in one sentence."}], "max_tokens": a.max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
hdr = {"Content-Type": "application/json"}
if a.api_key:
    hdr["Authorization"] = f"Bearer {a.api_key}"
req = urllib.request.Request(a.url + "/chat/completions", data=json.dumps(body).encode(), headers=hdr)
t0 = time.time()
first = None
n = 0
usage = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for c in ev.get("choices", []):
            if c.get("delta", {}).get("content"):
                if first is None:
                    first = time.time()
                n += 1
t1 = time.time()
pt = usage["prompt_tokens"] if usage else -1
ct = usage["completion_tokens"] if usage else n
ttft = (first or t1) - t0
dec = t1 - (first or t1)
decode = f", then {ct} tokens in {dec:.1f} s ({(ct - 1) / dec:.1f} tok/s decode)" if ct > 1 and dec > 0.05 else f", then {ct} tokens (arrived in one burst: no decode rate)"
print(f"prompt tokens {pt}, TTFT {ttft:.1f} s ({pt / ttft:.0f} tok/s prefill){decode}")
