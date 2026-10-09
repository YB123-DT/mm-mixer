#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_20261009
DEST=results/concat_mlp_20261009/meld_seed2025
mkdir -p "$DEST"
ssh biggpu /data2/yb/reproduction_envs/s0/bin/python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);run=root/'runs/meld_seed2025'
r=json.loads((run/'result.json').read_text())
assert r['status']=='completed' and r['epochs_completed']==50 and r['fresh_strict_replay_exact'] and not r['smoke_only']
assert (root/'exit_code.txt').read_text().strip()=='0'
a=np.load(run/'predictions.npz')
(run/'predictions.json').write_text(json.dumps({k:a[k].tolist() for k in a.files})+'\n')
PY
for f in result.json config.json environment.json data_manifest.json verification.json metrics.jsonl predictions.json predictions.npz; do
 scp "biggpu:$ROOT/runs/meld_seed2025/$f" "$DEST/$f"
done
scp "biggpu:$ROOT/exit_code.txt" "$DEST/exit_code.txt"
