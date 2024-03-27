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

import os
import unittest
from pathlib import Path
import random

import numpy as np

from aer_datasets.datasets.ascertain import AscertainDataset
from aer_datasets.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_MEDIA_FILES, \
    ASCERTAIN_NUM_PARTICIPANTS, ASCERTAIN_RAW_FOLDER


class AscertainDatasetTest(unittest.TestCase):
    def setUp(self):
        self.ecg_dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        self.ecg_dataset.load_trials()
        self.dataset_path = (DEFAULT_ASCERTAIN_PATH / ASCERTAIN_RAW_FOLDER).resolve()

    def test_ascertain_paths(self):
        """
        Asserts that the expected paths for the ASCERTAIN dataset exist...
        :return:
        """
        dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH)
        dataset.load_trials()

        for signal in dataset.signals:
            self.assertTrue(os.path.isdir(os.path.join(self.dataset_path, f'{signal}Data')))

        for path in np.array(sorted(os.listdir(self.dataset_path))):
            if os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, path)):
                if path.endswith('Data'):
                    self.assertIn(path.replace("Data", ""), dataset.signals)

    def test_ascertain_dataset_load(self):
        """
        Asserts that the ASCERTAIN dataset loads the expected number of participants, movie clips, and trials.
        :return:
        """
        self.assertEqual(len(self.ecg_dataset.participant_ids), ASCERTAIN_NUM_PARTICIPANTS)
        self.assertEqual(len(self.ecg_dataset.media_ids), ASCERTAIN_NUM_MEDIA_FILES)
        self.assertEqual(len(self.ecg_dataset.trials), ASCERTAIN_NUM_MEDIA_FILES * ASCERTAIN_NUM_PARTICIPANTS)

    def test_ascertain_path_limit_signals(self):
        """
        Asserts that the ASCERTAIN data set class works properly on a restricted list of signal types, using the
        ecg_dataset created in setUp
        :return:
        """
        self.assertTrue(True, 'We made it!')
        for signal in self.ecg_dataset.signals:
            self.assertTrue(os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, f'{signal}Data')))

        for path in np.array(sorted(os.listdir(self.dataset_path))):
            if os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, path)):
                if path.endswith('Data') and not path.startswith('ECG'):
                    self.assertNotIn(path.replace("Data", ""), self.ecg_dataset.signals)

        self.assertEqual(len(self.ecg_dataset.participant_ids), ASCERTAIN_NUM_PARTICIPANTS)
        self.assertEqual(len(self.ecg_dataset.media_ids), ASCERTAIN_NUM_MEDIA_FILES)

    @staticmethod
    def bad_signal():
        return AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['XYZ'])

    def test_ascertain_path_invalid_signal(self):
        """
        Asserts that a ValueError is thrown if AscertainDataset is constructed with a signal type that does not
        exist on the filesystem.
        :return:
        """
        self.assertRaises(ValueError, AscertainDatasetTest.bad_signal)

    def test_signal_datafiles(self):
        """
        Asserts that the all the datafiles loaded by the AscertainDataset are found.
        :return:
        """
        num_trials = 0
        num_data_files = 0
        for trial in self.ecg_dataset.trials:
            num_trials += 1
            for data_file in trial.signal_data_files.values():
                num_data_files += 1
                self.assertTrue(Path(data_file).exists(), msg=f"File {data_file} does not exist")
        self.assertEqual(num_trials, ASCERTAIN_NUM_PARTICIPANTS * ASCERTAIN_NUM_MEDIA_FILES)

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        trial = self.ecg_dataset.trials[random.randint(0, len(self.ecg_dataset.trials) - 1)]
        self.assertEqual(trial.load_signal_data('ECG').shape[0], 3)
