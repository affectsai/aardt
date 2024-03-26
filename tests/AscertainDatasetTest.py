import os
import unittest
from pathlib import Path
import random

import numpy as np

from datasets import AscertainDataset
from datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_MEDIA_FILES, \
    ASCERTAIN_NUM_PARTICIPANTS, ASCERTAIN_RAW_FOLDER


class AscertainDatasetTest(unittest.TestCase):
    def test_ascertain_path(self):
        dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH)
        dataset.load_trials()

        self.assertTrue(True, 'We made it!')
        for signal in dataset.signals:
            self.assertTrue(os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, f'{signal}Data')))

        for path in np.array(sorted(os.listdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER)))):
            if os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, path)):
                if path.endswith('Data'):
                    self.assertIn(path.replace("Data", ""), dataset.signals)

        self.assertEqual(len(dataset.participant_ids), ASCERTAIN_NUM_PARTICIPANTS)
        self.assertEqual(len(dataset.media_ids), ASCERTAIN_NUM_MEDIA_FILES)
        self.assertEqual(len(dataset.trials), ASCERTAIN_NUM_MEDIA_FILES * ASCERTAIN_NUM_PARTICIPANTS)

    def test_ascertain_path_limit_signals(self):
        dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        dataset.load_trials()

        self.assertTrue(True, 'We made it!')
        for signal in dataset.signals:
            self.assertTrue(os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, f'{signal}Data')))

        for path in np.array(sorted(os.listdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER.name)))):
            if os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, path)):
                if path.endswith('Data') and not path.startswith('ECG'):
                    self.assertNotIn(path.replace("Data", ""), dataset.signals)

        self.assertEqual(len(dataset.participant_ids), ASCERTAIN_NUM_PARTICIPANTS)
        self.assertEqual(len(dataset.media_ids), ASCERTAIN_NUM_MEDIA_FILES)

    @staticmethod
    def bad_signal():
        return AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['XYZ'])

    def test_ascertain_path_invalid_signal(self):
        self.assertRaises(ValueError, AscertainDatasetTest.bad_signal)

    def test_signal_datafiles(self):
        dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        dataset.load_trials()
        num_trials = 0
        num_data_files = 0
        for trial in dataset.trials:
            num_trials += 1
            for data_file in trial.signal_data_files.values():
                num_data_files += 1
                self.assertTrue(Path(data_file).exists(), msg=f"File {data_file} does not exist")
        self.assertEqual(num_trials, ASCERTAIN_NUM_PARTICIPANTS * ASCERTAIN_NUM_MEDIA_FILES)

    def test_ecg_signal_load(self):
        dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        dataset.load_trials()
        trial = dataset.trials[random.randint(0, len(dataset.trials) - 1)]
        self.assertEqual(trial.load_signal_data('ECG').shape[0], 3)
