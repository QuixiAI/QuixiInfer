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
