import unittest

from check_ds_log import status


GOOD = """1. exact f32 vs reference:      positions 6: top-1 agreement 6/6, KLD mean 0.000000
2. production vs reference:     positions 59: top-1 agreement 57/59, KLD mean 0.128643
3. windows of 3 vs single steps: bit-identical rows: 128 / 128
3. windows of 8 vs single steps: bit-identical rows: 128 / 128
3. windows of 64 vs single steps: bit-identical rows: 128 / 128
3b. windows of 3, tiers adapting, vs single steps: bit-identical rows: 128 / 128
4. tokenizer vs llama.cpp:      identical: 20 tokens decode round trip: exact 
5. rewind and park vs straight: rewind to 64: bit-identical rows: 64 / 64 park/unpark to 64: bit-identical rows: 64 / 64 
6. mixed sequences, cold widths: multi-sequence windows vs A alone: bit-identical rows: 3 / 3 multi-sequence slot 1 vs alone: bit-identical rows: 3 / 3 multi-sequence slot 2 vs alone: bit-identical rows: 3 / 3 
"""


class CheckLogTest(unittest.TestCase):
    def test_complete_current_suite(self):
        self.assertEqual(status(GOOD), "pass")

    def test_each_check_is_required_once(self):
        for line in GOOD.splitlines(keepends=True):
            with self.subTest(line=line):
                expected = "incomplete" if line.startswith("4.") else "fail"
                self.assertEqual(status(GOOD.replace(line, "")), expected)
                self.assertEqual(status(GOOD + line), "fail")
                self.assertEqual(status(GOOD + line.split(":", 1)[0] + ": WRONG\n"), "fail")

    def test_wrong_values_and_runtime_errors_fail(self):
        for old, new in [("128 / 128", "127 / 128"), ("128 / 128", "128 / 1280"),
                         ("57/59", "58/59"), ("0.128643", "0.128644"),
                         ("64 / 64", "63 / 64"), ("3 / 3", "2 / 3"),
                         ("decode round trip: exact", "decode round trip: wrong")]:
            with self.subTest(old=old, new=new):
                self.assertEqual(status(GOOD.replace(old, new, 1)), "fail")
        self.assertEqual(status(GOOD + "thread panicked: CUDA_ERROR_DEINITIALIZED\n"), "fail")


if __name__ == "__main__":
    unittest.main()
