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

from datetime import datetime, timedelta

import numpy as np
import scipy.io

from ardt.datasets import AERTrial

ASCERTAIN_ECG_SAMPLE_RATE = 256
ASCERTAIN_ECG_N_CHANNELS = 2


class AscertainTrial(AERTrial):
    def __init__(self, dataset, participant_id, movie_id, quadrant):
        super().__init__(dataset, participant_id, movie_id)
        self._ecg_signal_duration = None
        self._truth = quadrant

    def load_ground_truth(self):
        return self._truth

    def get_signal_metadata(self, signal_type):
        dataset_meta = self.dataset.get_signal_metadata(signal_type)
        if signal_type == 'ECG':
            dataset_meta['duration'] = self._ecg_signal_duration
        return dataset_meta

    def load_raw_signal_data(self, signal_type):
        signal_data_file = self._signal_data_files[signal_type]
        matfile = scipy.io.loadmat(signal_data_file)

        if signal_type == 'ECG':
            data = self._load_ecg_signal_data(matfile)
            self._ecg_signal_duration = data.shape[1] / ASCERTAIN_ECG_SAMPLE_RATE
            return data
        elif signal_type == 'GSR':
            return self._load_gsr_signal_data(matfile)
        elif signal_type == 'EEG':
            return self._load_eeg_signal_data(matfile)
        else:
            raise ValueError('_load_signal_data not implemented for signal type {}'.format(signal_type))

    @staticmethod
    def _load_eeg_signal_data(signal_data_file):
        return []

    @staticmethod
    def _load_ecg_signal_data(signal_data_file):
        """
        Loads the ECG signal data from the given ASCERTAIN matlab file, and returns it as a 3xN
        numpy array. The first row contains the signal timestamp data in UNIX Epoch time. The
        second and third rows contain the two ECG signal channels from the dataset.

        :param signal_data_file: The ECG_ClipXX.mat file for this trial
        :return:
        """
        start_time_arr = signal_data_file['timeECG'][0]
        start_time = datetime(
            int(start_time_arr[0]),
            int(start_time_arr[1]),
            int(start_time_arr[2]),
            int(start_time_arr[3]),
            int(start_time_arr[4]),
            int(start_time_arr[5]),
            int(1000 * (start_time_arr[5] % 1))
        )

        def convert_to_epoch(_timestamp, _start_time):
            return (_start_time + timedelta(milliseconds=_timestamp)).timestamp()

        timeconverter = np.vectorize(lambda _ts: convert_to_epoch(_ts, start_time))

        ecg_data = signal_data_file['Data_ECG']
        left_arm_idx = 1 if (len(ecg_data[0]) < 6) else 4
        right_arm_idx = 2 if (len(ecg_data[0]) < 6) else 5

        ecg = ecg_data[:, [0, left_arm_idx, right_arm_idx]]
        ts = np.apply_along_axis(
            func1d=timeconverter,
            axis=0,
            arr=ecg[:, 0])
        ts = ts.reshape(-1, 1)

        result = np.append(ts, ecg[:, [1, 2]], axis=1)
        return result.transpose()

    @staticmethod
    def _load_gsr_signal_data(signal_data_file):
        return []

    @property
    def participant_response(self):
        self.load_ground_truth()
