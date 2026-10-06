import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('baseline_queue', Path(__file__).with_name('queue.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_forbidden_physical_gpu(self):
        with patch.object(queue.subprocess, 'check_output', return_value='4, GPU-bad, 32768\n'):
            with self.assertRaisesRegex(RuntimeError, 'GPU 4'):
                queue.gpu_state(['GPU-bad'])

    def test_unknown_uuid_rejected(self):
        with patch.object(queue.subprocess, 'check_output', return_value='3, GPU-good, 32768\n'):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                queue.gpu_state(['GPU-other'])

    def test_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.py'
            path.write_text('original')
            digest = queue.hashlib.sha256(path.read_bytes()).hexdigest()
            job = {'source_sha256': {str(path): digest}}
            queue.verify_sources(job)
            path.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                queue.verify_sources(job)

    def test_other_seed_result_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = {'output': directory, 'model': 'Example', 'dataset': 'meld', 'seed': 2025}
            data = dict(status='completed', model='Example', dataset='meld', seed=2028, epochs_completed=15)
            (Path(directory) / 'result.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'seed'):
                queue.check_result(job)

    def test_smoke_cannot_complete_formal_job(self):
        with tempfile.TemporaryDirectory() as directory:
            job = {'output': directory, 'model': 'Example', 'dataset': 'meld', 'seed': 2025}
            data = dict(status='completed', model='Example', dataset='meld', seed=2025,
                        smoke_only=True, epochs_completed=15)
            (Path(directory) / 'result.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'smoke'):
                queue.check_result(job)


if __name__ == '__main__':
    unittest.main()
