#  Copyright (c) 2024-2025. Affects AI LLC
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

from aardt import config
from aardt.datasets import AERDataset

class MultiDataset(AERDataset):
    def __init__(self, datasets, signals=None):
        super().__init__(signals)
        self._datasets = datasets
        self._signal_metadata = {}

    def _preload_dataset(self):
        for dataset in self._datasets:
            dataset.preload()

    def load_trials(self):
        for dataset in self._datasets:
            dataset.load_trials()
            self.trials.extend(dataset.trials)
            self.participant_ids.update(dataset.participant_ids)
            self.media_ids.update(dataset.media_ids)

    def set_signal_metadata(self, signal_type, metadata):
        self._signal_metadata[signal_type] = metadata

    def get_signal_metadata(self, signal_type):
        return self._signal_metadata[signal_type]