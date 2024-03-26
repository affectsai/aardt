import abc


class AERDataset(metaclass=abc.ABCMeta):
    def __init__(self, signals=None, participant_offset=0, mediafile_offset=0):
        if signals is None:
            signals = []
        self._signals = signals
        self._signal_preprocessors = {}
        self._participant_offset = participant_offset
        self._media_file_offset = mediafile_offset
        self._participant_ids = set()
        self._media_ids = set()
        self._all_trials = []

    @abc.abstractmethod
    def load_trials(self):
        pass

    @property
    def signals(self):
        return self._signals

    @property
    def trials(self):
        return self._all_trials

    @property
    def media_ids(self):
        return self._media_ids

    @property
    def participant_ids(self):
        return self._participant_ids

    @property
    def media_file_offset(self):
        return self._media_file_offset

    @property
    def participant_offset(self):
        return self._participant_offset

    @property
    def all_trails(self):
        return self._all_trials

    @property
    def signal_preprocessors(self):
        return self._signal_preprocessors

