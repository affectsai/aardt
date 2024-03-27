#  Copyright (c) 2024. Affects AI LLC
#
#  Licensed under the Creative Common CC BY-NC-SA 4.0 International License (the "License");
#  you may not use this file except in compliance with the License. The full text of the License is
#  provided in the included LICENSE file. If this file is not available, you may obtain a copy of the
#  License at
#
#       https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing permissions and limitations
#  under the License.


import abc
from pathlib import Path

import numpy as np

from aer_datasets import config


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

    def preload(self):
        preload_file = self.get_working_dir() / Path('.preload.npy')
        if preload_file.exists():
            preloaded_signals = set(np.load(preload_file))

            # If self.signals is a subset of the signals that have already been preloaded
            # then we don't have to preload anything.
            if set(self.signals).issubset(preloaded_signals):
                return

        self._preload_dataset()
        np.save(preload_file, self.signals)

    @abc.abstractmethod
    def _preload_dataset(self):
        pass

    @abc.abstractmethod
    def load_trials(self):
        pass

    def get_working_dir(self):
        path = Path(config['working_dir'])/Path(self.__class__.__name__)
        path.mkdir(parents=True, exist_ok=True)
        return path

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
