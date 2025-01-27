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
import logging
from pathlib import Path

import scipy

from aardt import config
from aardt.datasets import AERDataset
from .AscertainTrial import AscertainTrial

CONFIG = config['datasets']['ascertain']
DEFAULT_ASCERTAIN_PATH = Path(CONFIG['path'])
ASCERTAIN_RAW_FOLDER = Path(CONFIG['raw_data_path'])
ASCERTAIN_FEATURES_FOLDER = Path(CONFIG['features_data_path'])
ASCERTAIN_NUM_MEDIA_FILES = 36
ASCERTAIN_NUM_PARTICIPANTS = 58

logger = logging.getLogger('AscertainDataset')
logger.level = logging.DEBUG


class AscertainDataset(AERDataset):
    def __init__(self, ascertain_path=None, signals=None, participant_offset=0, mediafile_offset=0):
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

        if ascertain_path is None:
            ascertain_path = CONFIG.get('path')

        if ascertain_path is None or not os.path.exists(ascertain_path):
            raise ValueError(
                f'Invalid path to ASCERTAIN dataset: {ascertain_path}. Please correct and try again.')

        logger.debug(f'Loading ASCERTAIN from {ascertain_path} with signals {signals}.')

        self.ascertain_path = Path(ascertain_path)
        self.ascertain_raw_path = self.ascertain_path / ASCERTAIN_RAW_FOLDER
        self.ascertain_features_path = self.ascertain_path / ASCERTAIN_FEATURES_FOLDER
        self.media_index_to_name = {}           # Maps media index back to name

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

        self._expected_results = {
            # Taken from DECAF+ mediafile_offset: MEG-BASED MULTIMODAL DATABASE FOR DECODING AFFECTIVE PHYSIOLOGICAL RESPONSES
            # Amusing
            1+ mediafile_offset: 1,   # Ace Ventura+ mediafile_offset: Pet Detective
            2+ mediafile_offset: 1,   # The Gods Must Be Crazy II
            4+ mediafile_offset: 1,   # Airplane
            5+ mediafile_offset: 1,   # When Harry Met Sally

            # Funny:
            3+ mediafile_offset: 1,   # Liar Liar
            6+ mediafile_offset: 1,   # The Gods Must Be Crazy
            7+ mediafile_offset: 1,   # The Hangover
            9+ mediafile_offset: 1,   # Hot Shots

            # Happy:
            8+ mediafile_offset: 4,   # Up
            10+ mediafile_offset: 4,  # August Rush
            11+ mediafile_offset: 4,  # Truman Show
            12+ mediafile_offset: 4,  # Wall-E
            13+ mediafile_offset: 4,  # Love Actually
            14+ mediafile_offset: 4,  # Remember the Titans
            16+ mediafile_offset: 4,  # Life is Beautiful
            17+ mediafile_offset: 4,  # Slumdog Millionaire
            18+ mediafile_offset: 4,  # House of Flying Daggers

            # Exciting
            15+ mediafile_offset: 1,  # Legally Blonde
            33+ mediafile_offset: 1,  # The Untouchables

            # Angry
            19+ mediafile_offset: 3,  # Ghandi
            21+ mediafile_offset: 3,  # Lagaan
            23+ mediafile_offset: 3,  # My Bodyguard
            35+ mediafile_offset: 3,  # Crash

            # Disgusting
            28+ mediafile_offset: 2,  # The Exorcist
            34+ mediafile_offset: 2,  # Pink Flamingos

            # Fear:
            30+ mediafile_offset: 2,  # The Shining
            36+ mediafile_offset: 2,  # Black Swan

            # Sad
            20+ mediafile_offset: 3,  # My Girl
            22+ mediafile_offset: 3,  # Bambi
            24+ mediafile_offset: 3,  # Up
            25+ mediafile_offset: 3,  # Life is Beautiful
            26+ mediafile_offset: 3,  # Remember the Titans
            27+ mediafile_offset: 3,  # Titanic
            31+ mediafile_offset: 3,  # Prestige

            # Shock
            29+ mediafile_offset: 2,  # Mulholland Drive
            32+ mediafile_offset: 2,  # Alien
        }

    def _preload_dataset(self):
        pass

    def load_trials(self):
        # Load trial data...
        all_trials = {}
        dt_selfreports_path = os.path.join(self.ascertain_features_path, "Dt_SelfReports.mat")
        dt_selfreports = scipy.io.loadmat(dt_selfreports_path)

        for matlab_file in self.ascertain_raw_path.rglob("*Clip*.mat"):
            movie_folder = matlab_file.parents[0].name
            signal_folder = matlab_file.parents[1].name

            signal_type = signal_folder.replace("Data", "")
            if signal_type not in self._signals:
                continue

            participant_id = int(movie_folder.split("_P")[1]) + self.participant_offset
            movie_id = int(matlab_file.name.upper().replace(f'{signal_type}_CLIP', '').replace('.MAT', ''))
            movie_id += self.media_file_offset

            self.media_index_to_name[movie_id] = movie_id   # no names, just ids... 1:1 map
            if participant_id not in all_trials.keys():
                all_trials[participant_id] = {}

            if movie_id not in all_trials[participant_id].keys():
                all_trials[participant_id][movie_id] = {}

            all_trials[participant_id][movie_id][signal_type] = matlab_file.resolve()

        def _to_quadrant(a,v):
            q=-1
            if a >= 3:  # A is high
                if v >= 0:  # A is High, V is Neg = Quad 0
                    q = 1
                else:
                    q = 2
            else:
                if v < 0:
                    q = 3
                else:
                    q = 4
            return q

        for participant_id in all_trials:
            self.participant_ids.add(participant_id)
            for movie_id in all_trials[participant_id]:
                self.media_ids.add(movie_id)
                arousal = dt_selfreports['Ratings'][0][participant_id - 1 - self.participant_offset][movie_id - 1 - self.media_file_offset]
                valence = dt_selfreports['Ratings'][1][participant_id - 1 - self.participant_offset][movie_id - 1 - self.media_file_offset]

                trial = AscertainTrial(self, participant_id, movie_id, _to_quadrant(arousal, valence))
                trial.signal_data_files = all_trials[participant_id][movie_id]
                trial.signal_preprocessors = self.signal_preprocessors
                self.trials.append(trial)

    def get_signal_metadata(self, signal_type):
        if signal_type == 'ECG':
            return {
                'signal_type': signal_type,
                'sample_rate': 256,
                'n_channels': 2,
            }

    @property
    def media_names_by_movie_id(self):
        return self.media_index_to_name

    @property
    def expected_media_responses(self):
        return self._expected_results