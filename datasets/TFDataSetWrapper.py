import tensorflow as tf

from . import AERTrial
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
