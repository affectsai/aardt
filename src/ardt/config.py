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
import os.path
import yaml

default_config = {
    'working_dir': '/mnt/affectsai/aerds/',
    'datasets': {
        'ascertain': {
            'path': '/mnt/affectsai/datasets/ascertain',
            'raw_data_path': 'ASCERTAIN_Raw',
            'features_data_path': 'ASCERTAIN_Features'
        },
        'dreamer': {
            'path': '/mnt/affectsai/datasets/dreamer',
            'dreamer_data_filename': "DREAMER_Data.json"
        },
        'cuads': {
            'path': '/mnt/affectsai/datasets/cuads',
        }
    },
}

user_config = None
if os.path.exists('ardt_config.yaml'):
    with open('ardt_config.yaml', 'r') as f:
        user_config = yaml.safe_load(f)

config = user_config if user_config is not None else default_config