# Decision-fidelity company launch audit

Status: operational failure record, 2026-07-30. No research row completed in
either failure described here, so these events provide no evidence for or
against N0/N1.

| Node | Stage reached | Failure | Data impact | Fix |
| --- | --- | --- | --- | --- |
| `run338950-sam4-6` / Node B | environment ready; queue preflight | Escaped quotes inside a `python3 -c` f-string caused a `SyntaxError`. | Queue stopped before starting a row; no marker. | `2185fdd` replaces the one-liner with parsed heredoc Python. |
| `run338936-sam4-5` / Node A | environment and queue preflight ready; first N0 row entered trial 20 | The anchored warm-up read local `pred_imgs` before its first assignment. All four shards raised `UnboundLocalError` at 0/300 steps. | No completed shard and no shared marker; the same row name is safe to resume. | `a184503` uses the already initialized `last_obs_gt` history. |

Both nodes passed the intended company environment check: Python 3.10,
PyTorch 2.4 NGC build, CUDA 12.5, OpenCV headless 4.10, Pymunk 7.3, W&B
0.22.3, and four visible GPUs.

Validation for `a184503`:

- Python compilation of `gpc_rank_eval.py`;
- 8 relevant company launcher/environment/summary tests;
- source audit confirming the warm-up predicate cannot reference `pred_imgs`
  before assignment.

Rerun policy: pull a commit containing `a184503` and rerun the same queue
command. Do not delete output directories or markers; the launcher resumes
completed shards and shared markers if any exist.
