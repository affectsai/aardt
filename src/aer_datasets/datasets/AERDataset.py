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
