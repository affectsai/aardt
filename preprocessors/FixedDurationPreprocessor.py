from abc import ABCMeta, abstractmethod

import numpy as np

from preprocessors import SignalPreprocessor


class FixedDurationPreprocessor(SignalPreprocessor):
    def __init__(self, signal_duration=45, sample_rate=256, padding_value=0, parent_preprocessor=None):
        """
        Preprocesses the signal to a fixed duration. If signal is less than signal_duration, it will be padded on the
        left with the padding_value.

        :param signal_duration: target signal length, in seconds
        :param sample_rate: target signal sample rate in Hz
        """
        super().__init__(parent_preprocessor)
        self.signal_duration = signal_duration
        self.sample_rate = sample_rate
        self.padding_value = padding_value

    def process_signal(self, signal):
        """

        :param signal: The signal to trim, with size NxM where N is the number of channels, and M is the number of samples.
        :param args:
        :param kwargs:
        :return:
        """
        num_channels = signal.shape[0]
        num_samples = signal.shape[1]
        target_samples = self.signal_duration * self.sample_rate

        if num_samples >= target_samples:
            return signal[:, np.arange(num_samples-target_samples, num_samples)]
        else:
            return np.concatenate([np.zeros((num_channels, target_samples-num_samples)), signal], axis=1)
