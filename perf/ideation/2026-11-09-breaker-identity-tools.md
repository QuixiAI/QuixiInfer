# Identity tooling review, 2026-11-09

Scope: the uncommitted `qx-ds-multi` and `compare_dumps.py` changes on 61679b2. Diagnostic tools only;
no performance claim, production engine change, or baseline promotion.

The independent breaker found two pre-existing comparator false passes: concatenating reasoning with answer
text hides changes at their boundary, and treating a missing text field as empty accepts malformed records.
Both are fixed and covered by CPU tests. Extra keys, extra occurrences, missing occurrences and empty dumps
also fail. Completion order across different prompts remains irrelevant.

The initial checker extension only moved A among the first and last two rows, leaving B's prompt rows unchecked.
The follow-up `QX_DS_MULTI_CHECK_ALL=1` compares every B and C row with its own slot's single-step reference.
Rotation deliberately preserves each slot's token order; it is not an exhaustive permutation test.

On the final tooling diff, the breaker reported **"No new correctness findings"**. Its read-only schedule
simulation covered 60 combinations of widths, lengths and rotation: per-slot positions and final counts matched.
It also checked token-slice/context bounds and that the asynchronous driver is joined before engine buffers are
dropped. GPU validation is recorded separately in the notebook; this review is not a PROMOTE verdict.

Follow-up review covered the `warm_width` pending-buffer restoration and the named ds-checks log validator.
The breaker found that successful-pattern counts still accepted a conflicting failing duplicate; the validator
now counts label occurrences as well, and tests append a conflicting duplicate of every check.

For the engine fix (prefetch disabled), the breaker reported **"No concrete correctness defect found"**:
all slots' pending allocations are saved byte-for-byte; compute is synchronized before snapshot and restoration;
upload completes before the real pass; ring capacity protects earlier history, and completed store rows are
recomputed before use. Prefetch-enabled dry-pass publication remains outside this result. The cold all-row
reproducer subsequently passed A 26/26, B 834/834 and C 26/26; full validation is recorded in the notebook.
