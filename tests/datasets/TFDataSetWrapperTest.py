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

from aer_datasets.datasets import TFDataSetWrapper
from aer_datasets.datasets import AscertainDataset

from aer_datasets.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_MEDIA_FILES, \
    ASCERTAIN_NUM_PARTICIPANTS
from aer_datasets.preprocessors import FixedDurationPreprocessor


class TFDataSetWrapperTest(unittest.TestCase):
    def setUp(self):
        self.dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        self.dataset.signal_preprocessors['ECG'] = FixedDurationPreprocessor(45, 56, 0)
        self.dataset.load_trials()

    def test_ascertain_dataset(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(1, 10)
        tfdsw = TFDataSetWrapper(dataset=self.dataset)
        tfds = tfdsw(batch_size=64, buffer_size=500, repeat=repeat_count)

        iteration = 0
        total_elems = 0

        # loop over the provided number of steps
        for batch in tfds:
            iteration += 1
            total_elems += len(batch[0])

        # stop the timer
        # return the difference between end and start times
        self.assertGreater(iteration, 0)
        self.assertEqual(ASCERTAIN_NUM_PARTICIPANTS * ASCERTAIN_NUM_MEDIA_FILES * repeat_count, total_elems)
