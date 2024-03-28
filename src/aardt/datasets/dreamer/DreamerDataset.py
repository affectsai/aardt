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

import logging
import ijson
import numpy as np
from pathlib import Path
from aardt import config

from aardt.datasets import AERDataset
from .DreamerTrial import DreamerTrial

CONFIG = config['datasets']['dreamer']
DEFAULT_DREAMER_PATH = Path(CONFIG['path'])
DEFAULT_DREAMER_FILENAME = Path(CONFIG['dreamer_data_filename'])
DREAMER_NUM_MEDIA_FILES = 18
DREAMER_NUM_PARTICIPANTS = 23
DREAMER_ALL_SIGNALS = {'ECG', 'EEG'}

logger = logging.getLogger('DreamerDataset')
logger.level = logging.DEBUG


class DreamerDataset(AERDataset):
    def __init__(self, dataset_path, signals=None, participant_offset=0, mediafile_offset=0,
                 dataset_fname=DEFAULT_DREAMER_FILENAME):
        """
        Construct a new DreamerDataset object using the given path.

        :param dataset_path: Path to the folder containing the dataset_fname file.
        :param signals: A list of signals to load, e.g. ['ECG','EEG'] to load ECG and EEG data. If None, all available
        signals are loaded.
        :param participant_offset: Constant value added to each participant identifier within this dataset. For example,
        if participant_offset is 32, then Participant 1 from this dataset's raw data will be returned as Participant 33.
        :param mediafile_offset: Constant value added to each media identifier within this dataset. For example, if
        mediafile_offset is 12, then Movie 1 from this dataset's raw data will be reported as Media ID 13.
        """
        if signals is None:
            signals = list(DREAMER_ALL_SIGNALS)
        else:
            for signal in signals:
                if signal not in DREAMER_ALL_SIGNALS:
                    raise ValueError(
                        f'{signal} signal does not exist in DREAMER. Please correct and try again.')

        super().__init__(signals, participant_offset, mediafile_offset)
        logger.debug(f'Loading DREAMER from {dataset_path}/{dataset_fname} with signals {signals}.')
        self._dataset_file = Path(dataset_path) / Path(dataset_fname)

        if not self._dataset_file.exists():
            raise ValueError('Path to DREAMER dataset does not exist: {}'.format(self._dataset_file.resolve()))

    def get_working_path(self, participant_id=None, media_id=None, signal_type=None, stimuli=True):
        if media_id is not None and participant_id is None:
            raise ValueError('participant_id must be given if media_id is specified.')

        if signal_type is not None and media_id is None:
            raise ValueError('media_id must be given if signal_type is specified.')

        if signal_type is not None and signal_type not in self.signals:
            raise ValueError('Invalid signal type: {}'.format(signal_type))

        result = self.get_working_dir()
        if participant_id is not None:
            result /= f'Participant_{participant_id-self.participant_offset:02d}'
            if media_id is not None:
                result /= f'Media_{media_id-self.media_file_offset:02d}'
                if signal_type is not None:
                    result /= f'{signal_type}_{"stimuli" if stimuli else "baseline"}.npy'

        return result

    def _preload_dataset(self):
        participant_id = self.participant_offset
        with open(self._dataset_file, 'rb') as f:
            participant_entries = ijson.items(f, 'item')
            for participant_entry in participant_entries:
                participant_id += 1
                participant_path = self.get_working_path(participant_id)
                participant_path.mkdir(parents=True, exist_ok=True)

                for signal in self.signals:
                    baseline_signal_data = participant_entry[signal]['baseline']
                    stimuli_signal_data = participant_entry[signal]['stimuli']

                    np.save(participant_path / Path('arousal.npy'), participant_entry['ScoreArousal'])
                    np.save(participant_path / Path('valence.npy'), participant_entry['ScoreValence'])

                    for c in range(DREAMER_NUM_MEDIA_FILES):
                        media_id = self.media_file_offset + c + 1
                        media_path = self.get_working_path(participant_id, media_id)
                        media_path.mkdir(parents=True, exist_ok=True)

                        np.save(media_path / Path(f'{signal}_stimuli.npy'), stimuli_signal_data[c])
                        np.save(media_path / Path(f'{signal}_baseline.npy'), baseline_signal_data[c])

    def load_trials(self):
        for p in range(DREAMER_NUM_PARTICIPANTS):
            p += 1
            participant_id = p + self.participant_offset
            self.participant_ids.add(participant_id)
            for c in range(DREAMER_NUM_MEDIA_FILES):
                c += 1
                media_id = c + self.media_file_offset
                self.media_ids.add(c + self.media_file_offset)
                trial = DreamerTrial(self, participant_id, media_id)
                trial.signal_preprocessors = self.signal_preprocessors
                for signal in self.signals:
                    trial.signal_types.add(signal)
                    trial.signal_data_files[signal] = self.get_working_path(participant_id, media_id, signal)
                self.all_trails.append(trial)
