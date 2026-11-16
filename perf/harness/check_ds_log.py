#!/usr/bin/env python3
"""Validate every named ds-checks result; missing tokenizer alone is incomplete."""
from pathlib import Path
import re
import sys


def status(text):
    expectations = [
        r"1\. exact f32 vs reference:\s+positions 6: top-1 agreement 6/6, KLD mean 0\.000000",
        r"2\. production vs reference:\s+positions 59: top-1 agreement 57/59, KLD mean 0\.128643",
        *(rf"3\. windows of {w} vs single steps:\s+bit-identical rows: 128 / 128" for w in (3, 8, 64)),
        r"3b\. windows of 3, tiers adapting, vs single steps:\s+bit-identical rows: 128 / 128",
        r"5\. rewind and park vs straight:\s+rewind to 64: bit-identical rows: 64 / 64\s+park/unpark to 64: bit-identical rows: 64 / 64",
        r"6\. mixed sequences, cold widths:\s+multi-sequence windows vs A alone: bit-identical rows: 3 / 3\s+multi-sequence slot 1 vs alone: bit-identical rows: 3 / 3\s+multi-sequence slot 2 vs alone: bit-identical rows: 3 / 3",
    ]
    for pattern in expectations:
        label = pattern.split(r":\s+", 1)[0]
        if (len(re.findall(r"^" + label + ":", text, re.M)) != 1
                or len(re.findall(r"^" + pattern + r"[ \t]*$", text, re.M)) != 1):
            return "fail"
    # part c (a rewind replayed through one exact chunk): judged when its lines are there (parts a and b alone stay valid)
    if re.search(r"^7\.", text, re.M):
        for n, p, rows in ((560, 400, 160), (1280, 1024, 256)):
            line = rf"^7\. rewind to {p} of {n} by a replay chunk:\s+rewind to {p}: bit-identical rows: {rows} / {rows}[ \t]*$"
            if len(re.findall(line, text, re.M)) != 1:
                return "fail"
        if len(re.findall(r"^7\.", text, re.M)) != 2:
            return "fail"
    # the ring checkpoint and the rewind after a rewind (2026-11-15): judged when there
    if re.search(r"^[89]\.", text, re.M):
        for line in (r"8\. checkpoint at 420 of 560, rewind to 400:\s+rewind to 400 \(free\): bit-identical rows: 160 / 160",
                     r"8\. checkpoint at 420 of 600, rewind to 400, 10 tokens, rewind to 300:\s+rewind to 300 \(free\): bit-identical rows: 300 / 300",
                     r"9\. rewind to 400 after a rewind to 500 and 20 tokens, of 600:\s+rewind to 400 \(replay\): bit-identical rows: 200 / 200"):
            if len(re.findall(r"^" + line + r"[ \t]*$", text, re.M)) != 1:
                return "fail"
        if len(re.findall(r"^[89]\.", text, re.M)) != 3:
            return "fail"
    if re.search(r"panic|error", text, re.I):
        return "fail"
    tokenizer = r"^4\. tokenizer vs llama\.cpp:\s+identical: 20 tokens\s+decode round trip: exact[ \t]*$"
    if len(re.findall(r"^4\.", text, re.M)) == 1 and len(re.findall(tokenizer, text, re.M)) == 1:
        return "pass"
    return "fail" if re.search(r"^4\.", text, re.M) else "incomplete"


if __name__ == "__main__":
    result = status(Path(sys.argv[1]).read_text())
    print(result)
    sys.exit(0 if result == "pass" else 1)
