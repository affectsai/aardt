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
import numpy as np

from aardt.datasets import AERTrial

DREAMER_ECG_SAMPLE_RATE = 256
DREAMER_ECG_N_CHANNELS = 2


class DreamerTrial(AERTrial):
    def __init__(self, dataset, participant_id, movie_id):
        super().__init__(dataset, participant_id, movie_id)

    def load_signal_data(self, signal_type):
        signal = np.load(self.dataset.get_working_path(self.participant_id, self.movie_id, signal_type))
        time_steps = (np.arange(0, signal.shape[0]) * 1000 / 256).reshape(-1, 1)
        result = np.append(time_steps, signal, axis=1)

        return result.transpose()

    def load_ground_truth(self):
        return 0

    def get_signal_metadata(self, signal_type):
        dataset_meta = self.dataset.get_signal_metadata(signal_type)
        return dataset_meta
