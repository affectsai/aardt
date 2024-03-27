import unittest

import numpy as np

from preprocessors import FixedDurationPreprocessor

# Test parameters
LONG_SIGNAL_DURATION = 10
SHORT_SIGNAL_DURATION = 7
SAMPLE_RATE = 256


class FixedDurationPreprocessorTest(unittest.TestCase):
    def test_fixed_duration_preprocessor_long_signal(self):
        """
        Tests that when given a signal longer than the target duration, that the signal is truncated to the target
        duration, and that the truncated signal is from the tail end of the input signal.
        """
        signal = np.random.random(size=(3, SAMPLE_RATE * LONG_SIGNAL_DURATION))
        preprocessor = FixedDurationPreprocessor(signal_duration=SHORT_SIGNAL_DURATION, sample_rate=SAMPLE_RATE,
                                                 padding_value=0)
        processed = preprocessor(signal)

        target_num_samples = SAMPLE_RATE*(LONG_SIGNAL_DURATION-(LONG_SIGNAL_DURATION-SHORT_SIGNAL_DURATION))
        signal_num_samples = SAMPLE_RATE*LONG_SIGNAL_DURATION
        trimmed_signal = signal[:, np.arange(signal_num_samples-target_num_samples, signal_num_samples)]

        # Assert that the processed signal has the expected number of samples
        self.assertEqual(SAMPLE_RATE * SHORT_SIGNAL_DURATION, processed.shape[1])

        # Assert that expected output minus the processed output is zero for all values
        self.assertFalse((trimmed_signal-processed).all())

    def test_fixed_duration_preprocessor_short_signal(self):
        """
        Tests that when given a signal shorter than the target duration, that the signal is padded to the target
        duration, and that the padded values appear at the start of the processed signal
        """
        signal = np.random.random(size=(3, SAMPLE_RATE * SHORT_SIGNAL_DURATION))
        preprocessor = FixedDurationPreprocessor(signal_duration=LONG_SIGNAL_DURATION, sample_rate=SAMPLE_RATE,
                                                 padding_value=0)
        processed = preprocessor(signal)

        # Extract the values we expect to be padding...
        padded_values = processed[:, np.arange(0, LONG_SIGNAL_DURATION-SHORT_SIGNAL_DURATION)]

        # Assert that the processed signal has the expected number of samples
        self.assertEqual(SAMPLE_RATE * LONG_SIGNAL_DURATION, processed.shape[1])

        # Assert that the expected padding values are all zero.
        self.assertFalse(padded_values.all())

    def test_fixed_duration_preprocessor_equal_signal(self):
        """
        Tests that when given a signal that is has the target number of samples, that the signal is returned
        unmodified.
        """
        signal = np.random.random(size=(3, SAMPLE_RATE * LONG_SIGNAL_DURATION))
        preprocessor = FixedDurationPreprocessor(signal_duration=LONG_SIGNAL_DURATION, sample_rate=SAMPLE_RATE,
                                                 padding_value=0)
        processed = preprocessor(signal)
        self.assertFalse((signal-processed).all())