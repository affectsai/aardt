import unittest

import numpy as np

from preprocessors import FixedDurationPreprocessor


class FixedDurationPreprocessorTest(unittest.TestCase):
    def test_fixed_duration_preprocessor(self):
        SIGNAL_DURATION=10
        TARGET_DURATION=7
        SAMPLE_RATE=256

        signal = np.random.random(size=(3,SAMPLE_RATE*SIGNAL_DURATION))
        preprocessor = FixedDurationPreprocessor(signal_duration=TARGET_DURATION, sample_rate=SAMPLE_RATE, padding_value=0)
        processed = preprocessor(signal)

        self.assertEqual(SAMPLE_RATE*TARGET_DURATION, processed.shape[1])