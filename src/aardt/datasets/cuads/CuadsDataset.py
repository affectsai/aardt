#  Copyright (c) 2025. Affects AI LLC
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
import os
from pathlib import Path

from aardt import config
from aardt.datasets import AERDataset
from .CuadsTrial import CuadsTrial
import numpy as np

CONFIG = config['datasets']['cuads']
DEFAULT_DATASET_PATH = Path(CONFIG['path'])
CUADS_NUM_MEDIA_FILES   = 20
CUADS_NUM_PARTICIPANTS  = 44     # There are only 38, but they're still numbered 1 to 44.

logger = logging.getLogger('CuadsDataset')
logger.level = logging.DEBUG


class CuadsDataset(AERDataset):
    def __init__(self, dataset_path=None, participant_offset=0, mediafile_offset=0):
        """
        Construct a new CuadsDataset object..

        :param path: Path to the extracted CUADS dataset
        :param signals: A list of signals to load, e.g. ['ECG','EEG'] to load ECG and EEG data. If None, the folder
        given by dataset_path will be checked for subfolders named '<SIGNAL>Data', and each one found will be loaded.
        For example, if dreamer/ECGData/ is found, the ECG is automatically loaded.
        :param participant_offset: Constant value added to each participant identifier within this dataset. For example,
        if participant_offset is 32, then Participant 1 from this dataset's raw data will be returned as Participant 33.
        :param mediafile_offset: Constant value added to each media identifier within this dataset. For example, if
        mediafile_offset is 12, then Movie 1 from this dataset's raw data will be reported as Media ID 13.
        """
        signals = ['ECG', 'PPG', 'GSR']
        super().__init__(signals, participant_offset, mediafile_offset)

        if dataset_path is None:
            dataset_path = DEFAULT_DATASET_PATH

        if not os.path.exists(dataset_path):
            raise ValueError(
                f'Invalid path to DREAMER dataset: {dataset_path}. Please correct and try again.')

        logger.info(f'Loading CUADS from {dataset_path} with signals {signals}.')
        self.media_index_map = {}
        self.dataset_path = Path(dataset_path)



    def _preload_dataset(self):
        pass

    def load_trials(self):
        response_movie_name = 0
        response_valence = 1
        response_arousal = 2

        def _to_quadrant(a, v):
            q = -1
            if a >= 5:
                if v >= 5:
                    q = 1
                else:
                    q = 2
            else:
                if v < 5:
                    q = 3
                else:
                    q = 4
            return q

        # Load trial data...
        all_trials = {}
        for p in range(CUADS_NUM_PARTICIPANTS):
            cuads_participant_number = p + 1
            dataset_participant_number = cuads_participant_number + self.participant_offset

            participant_id = f'CUADS_{cuads_participant_number:03}'
            participant_folder = os.path.join( self.dataset_path, participant_id )
            response_file = os.path.join( participant_folder, 'responses.csv' )
            if not os.path.exists(response_file):
                continue

            self.participant_ids.add(dataset_participant_number)
            if participant_id not in all_trials.keys():
                all_trials[participant_id] = {}

            # Load this participant's responses...
            responses = np.loadtxt(response_file, delimiter=',', dtype=str, skiprows=1)
            for response_number, response in enumerate(responses):
                movie_name = response[response_movie_name]
                segmented_data_filepath = os.path.join(participant_folder, 'segmented', f'{movie_name}_sessiondata.csv')

                if not os.path.exists(segmented_data_filepath):
                    continue

                if movie_name not in self.media_index_map:
                    self.media_index_map[movie_name] = len(self.media_index_map) + 1

                movie_id = self.media_index_map[movie_name] + self.media_file_offset
                self.media_ids.add(movie_id)
                self.trials.append(
                    CuadsTrial(segmented_data_filepath,
                               dataset_participant_number,
                               movie_id,
                               _to_quadrant(float(response[response_arousal]), float(response[response_valence])))
                    )


