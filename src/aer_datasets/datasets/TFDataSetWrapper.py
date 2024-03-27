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

from .AERDataset import AERDataset
from tensorflow.data import AUTOTUNE


class TFDataSetWrapper:
    def __init__(self, dataset: AERDataset):
        self._aer_dataset = dataset

    def trial_generator(self):
        for trial in self._aer_dataset.trials:
            yield (tf.constant(trial.load_preprocessed_signal_data('ECG'), dtype=tf.float32),
                   tf.constant(trial.load_ground_truth(), dtype=tf.int32))

    def __call__(self, batch_size=64, buffer_size=1000, repeat=1, *args, **kwargs):
        dataset = tf.data.Dataset.from_generator(self.trial_generator,
                                                 output_signature=(tf.TensorSpec((3, None), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(), dtype=tf.int32)))
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True) \
            .cache() \
            .repeat(count=repeat) \
            .batch(batch_size, num_parallel_calls=AUTOTUNE, deterministic=False) \
            .prefetch(AUTOTUNE)

        return dataset
