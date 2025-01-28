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
import numpy as np

from ardt.preprocessors import SignalPreprocessor


class ChannelSelector(SignalPreprocessor):
    """
    Use this to select specific channels from the signal. Useful for removing timestamp data, or narrowing down
    channels in use for high-channel data.
    """

    def __init__(self, retain_channels=None, parent_preprocessor=None, child_preprocessor=None):
        """

        :param retain_channels: an iterable containing the channels to retain in the signal. If None, all channels except 0 are retained, removing timeseries data from the trial
        :param parent_preprocessor
        """
        super().__init__(parent_preprocessor, child_preprocessor)
        self._retain_channels = retain_channels

    def process_signal(self, signal):
        channels_to_keep = self._retain_channels
        if channels_to_keep is None:
            channels_to_keep = np.arange(1, signal.shape[0])
        return signal[channels_to_keep, :]
