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

import unittest
import random

from aer_datasets.datasets.dreamer.DreamerDataset import DreamerDataset
from aer_datasets.datasets.dreamer.DreamerDataset import (DEFAULT_DREAMER_PATH, DEFAULT_DREAMER_FILENAME,
                                                          DREAMER_NUM_MEDIA_FILES, DREAMER_NUM_PARTICIPANTS)


class DreamerDatasetTest(unittest.TestCase):
    def setUp(self):
        self.ecg_dataset = DreamerDataset(DEFAULT_DREAMER_PATH, signals=['ECG'])
        self.ecg_dataset.preload()
        self.ecg_dataset.load_trials()
        self.dataset_path = (DEFAULT_DREAMER_PATH / DEFAULT_DREAMER_FILENAME).resolve()

    def test_dataset_load(self):
        """
        Asserts that the ASCERTAIN dataset loads the expected number of participants, movie clips, and trials.
        :return:
        """
        self.assertEqual(len(self.ecg_dataset.participant_ids), DREAMER_NUM_PARTICIPANTS)
        self.assertEqual(len(self.ecg_dataset.media_ids), DREAMER_NUM_MEDIA_FILES)
        self.assertEqual(len(self.ecg_dataset.trials), DREAMER_NUM_MEDIA_FILES * DREAMER_NUM_PARTICIPANTS)

    def test_dataset_preload_files_exist(self):
        for trial in self.ecg_dataset.trials:
            for signal_type in trial.signal_types:
                self.assertTrue(trial.signal_data_files[signal_type].exists())

    @staticmethod
    def bad_signal():
        return DreamerDataset(DEFAULT_DREAMER_PATH, signals=['XYZ'])

    def test_invalid_signal(self):
        """
        Asserts that a ValueError is thrown if DreamerDataset is constructed with a signal type that does not
        exist on the filesystem.
        :return:
        """
        self.assertRaises(ValueError, DreamerDatasetTest.bad_signal)

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        trial = self.ecg_dataset.trials[random.randint(0, len(self.ecg_dataset.trials) - 1)]
        self.assertEqual(trial.load_signal_data('ECG').shape[0], 3)
