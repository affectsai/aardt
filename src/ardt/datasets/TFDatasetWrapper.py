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


import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np

from .AERDataset import AERDataset


class TFDatasetWrapper:
    """
    A utility class that wraps an AERDataset in a tf.data.Dataset for use in model training.
    """

    def __init__(self, dataset: AERDataset, splits=None):
        self._aer_dataset = dataset
        self._splits = splits if splits is not None else [1]
        self._trial_splits = self._aer_dataset.get_trial_splits(self._splits)
        if len(self._splits) == 1:
            self._trial_splits = [self._trial_splits]

    def __call__(self, signal_type, batch_size=64, buffer_size=1000, repeat=None, n_split=0):
        """
        :param batch_size: the size of the batch to generate
        :param buffer_size: the number of trials to prefetch into memory
        :param repeat: the number of times this dataset should repeat over itself
        :param n_split: if splits were given, this specifies the index of the split to use.
        :return:
        """

        def _trial_generator():
            for trial in self._trial_splits[n_split]:
                label = trial.load_ground_truth()
                yield (tf.constant(trial.load_signal_data(signal_type).transpose()  , dtype=tf.float32),
                       tf.constant(np.array([0 if label is None else label]).reshape(-1,1), dtype=tf.int32))

        num_channels = self._aer_dataset.get_signal_metadata(signal_type)['n_channels']
        dataset = tf.data.Dataset.from_generator(_trial_generator,
                                                 output_signature=(tf.TensorSpec((None,num_channels), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(None,1), dtype=tf.int32)))

        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True) \
            .cache() \
            .repeat(repeat) \
            .batch(batch_size, num_parallel_calls=AUTOTUNE, deterministic=False) \
            .prefetch(AUTOTUNE)

        return dataset
