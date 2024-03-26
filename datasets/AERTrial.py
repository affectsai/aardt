import abc

import numpy as np


class AERTrial(abc.ABC):
    def __init__(self, participant_id, movie_id):
        self._participant_id = participant_id
        self._signal_types = set()
        self._signal_preprocessors = {}
        self._signal_data_files = {}
        self._movie_id = movie_id

    def load_preprocessed_signal_data(self, signal_type):
        signal_data = self.load_signal_data(signal_type)
        if signal_type in self.signal_preprocessors.keys():
            signal_data = self.signal_preprocessors[signal_type](signal_data)
        return signal_data

    @abc.abstractmethod
    def load_signal_data(self, signal_type):
        """
        Loads and returns the requested signal as an (N+1)xM numpy array, where N is the number of channels, and M is
        the number of samples in the signal. The row at N=0 represents the timestamp of each sample. The value is
        given in epoch time if a real start time is available, otherwise it is in elapsed milliseconds with 0
        representing the start of the sample.

        :param signal_type:
        :return:
        """
        if signal_type not in self._signal_types:
            raise ValueError('Signal type {} is not known in this AERTrial'.format(signal_type))

        return np.empty(0)

    @abc.abstractmethod
    def load_ground_truth(self):
        """
        Returns the ground truth label for this trial. For AER trials, this is the quadrant within the A/V space,
        numbered 0 through 3 as follows:
        - 0: High Arousal, High Valence
        - 1: High Arousal, Low Valence
        - 2: Low Arousal, Low Valence
        - 3: Low Arousal, High Valence

        :return: The ground truth label for this trial
        """
        return 0

    @abc.abstractmethod
    def get_signal_metadata(self, signal_type):
        """
        Returns a dict containing the requested signal's metadata. Mandatory keys include:
        - 'signal_type' (the signal type)
        - 'sample_rate' (in samples per second)
        - 'n_channels' (the number of channels in the signal)

        See subclasses for implementation-specific keys that may also be present.

        :param signal_type: the type of signal for which to retrieve the metadata.
        :return: a dict containing the requested signal's metadata
        """
        if signal_type not in self._signal_types:
            raise ValueError('Signal type {} is not known in this AERTrial'.format(signal_type))

        return {}

    @property
    def signal_data_files(self):
        return self._signal_data_files

    @signal_data_files.setter
    def signal_data_files(self, signal_data_files):
        for signal_type in signal_data_files.keys():
            self._signal_types.add(signal_type)

        self._signal_data_files = signal_data_files

    @property
    def signal_types(self):
        return self._signal_types

    @signal_types.setter
    def signal_types(self, signal_types):
        self._signal_types = signal_types

    @property
    def signal_preprocessors(self):
        return self._signal_preprocessors

    @signal_preprocessors.setter
    def signal_preprocessors(self, signal_preprocessors):
        self._signal_preprocessors = signal_preprocessors

    @property
    def movie_id(self):
        return self._movie_id

    @property
    def participant_id(self):
        return self._participant_id
