# MAGTKD reproduction

**Completed6/6 formal runs.** See [final results](../../../results/replacement_20261007/magtkd/README.md), [audit](AUDIT.md), and [fixed plan](plan.json).

`run.py` minimally wraps released stage-two training; `run_queue.py` persists six runs on healthy biggpu GPU1; `summarize.py` validates prediction-derived metrics; `fetch.py` downloads official ZIP feature members with CRC/SHA verification; `audit_data.py` verifies data shape/counts.

Current parameters/source/data/seed configurations are frozen in `plan.json`/`source_manifest.json`. `launch_state.json` is historical startup evidence. Final all-completed queue state and per-seed evidence live with the results. Do not call fixed first-stage feature reuse end-to-end three-seed training.
