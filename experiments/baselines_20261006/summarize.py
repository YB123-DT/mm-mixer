"""Verify prediction-derived baseline metrics and summarize all planned seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

# The controller's historical filename queue.py must not shadow stdlib queue
# when sklearn imports multiprocessing. This script needs no sibling imports.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    rows, groups = [], {}
    for job in plan['jobs']:
        root = Path(job['output'])
        row = {k: job[k] for k in ('id', 'model', 'dataset', 'seed')}
        row['output'] = str(root)
        try:
            result = json.loads((root / 'result.json').read_text())
            assert result['status'] == 'completed', result['status']
            assert all(result[k] == job[k] for k in ('model', 'dataset', 'seed'))
            with np.load(root / 'predictions.npz', allow_pickle=False) as pred:
                y, yp = pred['y_true'], pred['y_pred']
            assert len(y) == {'iemocap': 1623, 'meld': 2610}[job['dataset']], len(y)
            values = {'accuracy': accuracy_score(y, yp) * 100,
                      'weighted_f1': f1_score(y, yp, average='weighted') * 100,
                      'macro_f1': f1_score(y, yp, average='macro') * 100}
            for name, value in values.items():
                assert abs(value - result['test'][name]) < 1e-6, name
            row.update(status='verified', test=values, selected_epoch=result['selected_epoch'],
                       selection=result['selection'],
                       result_sha256=hashlib.sha256((root / 'result.json').read_bytes()).hexdigest(),
                       predictions_sha256=hashlib.sha256((root / 'predictions.npz').read_bytes()).hexdigest())
            groups.setdefault((job['model'], job['dataset']), []).append(row)
        except Exception as exc:
            row.update(status='unverified_or_missing', error=str(exc))
        rows.append(row)
    summary = []
    for (model, dataset), members in groups.items():
        planned = [j for j in plan['jobs'] if (j['model'], j['dataset']) == (model, dataset)]
        item = {'model': model, 'dataset': dataset, 'n': len(members), 'planned': len(planned)}
        if len(members) == len(planned) == 3:
            item['metrics'] = {key: {'mean': statistics.mean(r['test'][key] for r in members),
                                    'std_ddof1': statistics.stdev(r['test'][key] for r in members)}
                               for key in ('accuracy', 'weighted_f1', 'macro_f1')}
        summary.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'complete': all(r['status'] == 'verified' for r in rows),
                                     'runs': rows, 'summary': summary}, indent=2) + '\n')


if __name__ == '__main__':
    main()
