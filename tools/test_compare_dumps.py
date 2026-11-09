"""The server identity gate must reject incomplete or extra generations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CompareDumpsTest(unittest.TestCase):
    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("a.jsonl", "b.jsonl")]
            for path, rows in zip(paths, (a, b)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return subprocess.run(
                [sys.executable, "-I", str(Path(__file__).with_name("compare_dumps.py")), *map(str, paths)],
                capture_output=True, text=True,
            )

    def test_completion_order_does_not_matter(self):
        a = dict(c=3, prompt="A", text="one")
        b = dict(c=3, prompt="B", text="two")
        self.assertEqual(self.compare([a, b], [b, a]).returncode, 0)

    def test_mismatch_missing_extra_and_empty_fail(self):
        a = dict(c=3, prompt="A", text="one")
        b = dict(c=3, prompt="B", text="two")
        cases = [
            ([a], [dict(a, text="changed")]),
            ([a], [dict(a, reasoning="changed")]),
            ([dict(a, reasoning="ab", text="c")], [dict(a, reasoning="a", text="bc")]),
            ([dict(a, text="")], [dict(c=3, prompt="A")]),
            ([a], [dict(a, text=None)]),
            ([a, b], [a]),
            ([a], [a, b]),
            ([a], [a, a]),
            ([a, a], [a]),
            ([], []),
            ([], [a]),
            ([a], []),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                result = self.compare(left, right)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertNotIn("\nidentical\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
