#  Copyright (c) 2025. Affects AI LLC
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

from datetime import datetime, timedelta

import numpy as np
import scipy.io

from aardt.datasets import AERTrial

SAMPLE_RATE = 256

CUADS_COLUMN_MAP = {
    "SEGMENT_ECG_LARA":     15,
    "SEGMENT_ECG_LLLA":     16,
    "SEGMENT_ECG_LLRA":     17,
    "SEGMENT_ECG_HR_LARA":  19,
    "SEGMENT_ECG_HR_LLLA":  20,
    "SEGMENT_ECG_HR_LLRA":  21,
    "SEGMENT_GSR_SC":       37,
    "SEGMENT_GSR_SR":       38,
    "SEGMENT_PPG":          45,
    "SEGMENT_PPG_IBI":      46,
    "SEGMENT_PPG_HR":       47,
}

class CuadsTrial(AERTrial):
    def __init__(self, dataset, segment_file, participant_id, movie_id, truth):
        super().__init__(dataset, participant_id, movie_id)
        self._truth = truth
        self._segmented_file = segment_file
        self._trial_duration = 0

    def load_ground_truth(self):
        return self._truth

    def get_signal_metadata(self, signal_type):
        dataset_meta = self.dataset.get_signal_metadata(signal_type)
        dataset_meta['duration'] = self._trial_duration
        return dataset_meta


    def load_signal_data(self, signal_type):
        segment_data = np.loadtxt(self._segmented_file, delimiter=',', dtype=str, skiprows=1)
        self._trial_duration = segment_data.shape[1] / SAMPLE_RATE

        if signal_type == 'ECG':
            data = np.array( segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_ECG_LARA"],
                                              CUADS_COLUMN_MAP["SEGMENT_ECG_LLLA"],
                                              CUADS_COLUMN_MAP["SEGMENT_ECG_LLRA"]]], dtype=float)
            return data.transpose()
        elif signal_type == 'GSR':
            data = np.array( segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_GSR_SC"],
                                              CUADS_COLUMN_MAP["SEGMENT_GSR_SR"]]], dtype=float)
            return data.transpose()
        elif signal_type == 'PPG':
            data = np.array(segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_PPG"]]], dtype=float)
            return data.transpose()
        else:
            raise ValueError('load_signal_data not implemented for signal type {}'.format(signal_type))


