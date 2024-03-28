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
from pathlib import Path

from aardt.datasets import AERDataset
from .AscertainTrial import AscertainTrial
from aardt import config

CONFIG = config['datasets']['ascertain']
DEFAULT_ASCERTAIN_PATH = Path(CONFIG['path'])
ASCERTAIN_RAW_FOLDER = Path(CONFIG['raw_data_path'])
ASCERTAIN_NUM_MEDIA_FILES = 36
ASCERTAIN_NUM_PARTICIPANTS = 58

logger = logging.getLogger('AscertainDataset')
logger.level = logging.DEBUG


class AscertainDataset(AERDataset):
    def __init__(self, ascertain_path, signals=None, participant_offset=0, mediafile_offset=0):
        """
        Construct a new AscertainDataset object for a given ascertainPath.

        :param ascertain_path: Path to the extracted ASCERTAIN dataset
        :param signals: A list of signals to load, e.g. ['ECG','EEG'] to load ECG and EEG data. If None, the folder
        given by ascertain_path will be checked for subfolders named '<SIGNAL>Data', and each one found will be loaded.
        For example, if ascertain/ECGData/ is found, the ECG is automatically loaded.
        :param participant_offset: Constant value added to each participant identifier within this dataset. For example,
        if participant_offset is 32, then Participant 1 from this dataset's raw data will be returned as Participant 33.
        :param mediafile_offset: Constant value added to each media identifier within this dataset. For example, if
        mediafile_offset is 12, then Movie 1 from this dataset's raw data will be reported as Media ID 13.
        """
        super().__init__(signals, participant_offset, mediafile_offset)
        logger.debug(f'Loading ASCERTAIN from {ascertain_path} with signals {signals}.')

        self.ascertain_path = Path(ascertain_path)
        self.ascertain_raw_path = self.ascertain_path / ASCERTAIN_RAW_FOLDER

        if not self.ascertain_path.exists():
            raise ValueError('Path to ASCERTAIN does not exist: {}'.format(ascertain_path))

        # Load signals
        if signals is not None:
            for signal in signals:
                if not (self.ascertain_raw_path / f'{signal}Data').exists():
                    raise ValueError(
                        f'{signal}Data does not exist, unable to load {signal} Signal. Please correct and try again.')
        else:
            for p in sorted(self.ascertain_raw_path.rglob("*Data")):
                if p.is_dir():
                    self.signals.append(str(p.name).replace("Data", ""))

    def _preload_dataset(self):
        pass

    def load_trials(self):
        # Load trial data...
        all_trials = {}
        for matlab_file in self.ascertain_raw_path.rglob("*Clip*.mat"):
            movie_folder = matlab_file.parents[0].name
            signal_folder = matlab_file.parents[1].name

            signal_type = signal_folder.replace("Data", "")
            if signal_type not in self._signals:
                continue

            participant_id = int(movie_folder.split("_P")[1]) + self.participant_offset
            movie_id = int(matlab_file.name.upper().replace(f'{signal_type}_CLIP', '').replace('.MAT', ''))
            movie_id += self.media_file_offset

            if participant_id not in all_trials.keys():
                all_trials[participant_id] = {}

            if movie_id not in all_trials[participant_id].keys():
                all_trials[participant_id][movie_id] = {}

            all_trials[participant_id][movie_id][signal_type] = matlab_file.resolve()

        for participant_id in all_trials:
            self.participant_ids.add(participant_id)
            for movie_id in all_trials[participant_id]:
                self.media_ids.add(movie_id)
                trial = AscertainTrial(participant_id, movie_id)
                trial.signal_data_files = all_trials[participant_id][movie_id]
                trial.signal_preprocessors = self.signal_preprocessors
                self.all_trails.append(trial)
