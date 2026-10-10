from __future__ import annotations

import unittest

from mm_mixer_final.lr_schedules import warmup_cosine_decay


class WarmupCosineDecayTest(unittest.TestCase):
    def test_warmup_then_decay_is_monotonic_after_peak(self):
        total_steps = 100
        warmup_steps = 20
        values = [
            warmup_cosine_decay(step, total_steps, warmup_steps)
            for step in range(total_steps + 1)
        ]

        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[warmup_steps], 1.0)
        self.assertAlmostEqual(values[-1], 0.0)
        self.assertTrue(
            all(left >= right for left, right in zip(values[warmup_steps:], values[warmup_steps + 1 :]))
        )

    def test_steps_after_training_remain_at_zero(self):
        self.assertEqual(warmup_cosine_decay(120, 100, 20), 0.0)


if __name__ == "__main__":
    unittest.main()
