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

from abc import ABCMeta, abstractmethod


class SignalPreprocessor(metaclass=ABCMeta):
    def __init__(self, parent_preprocessor=None):
        """
        Constructs a signal preprocessor with an optional parent. If parent_preprocessor is given, then it will
        be called before this one, allowing for a preprocessor chain to be constructed

        :param parent_preprocessor:
        """
        self.parent_preprocessor = parent_preprocessor

    @abstractmethod
    def process_signal(self, signal):
        """

        :param signal: The signal to trim, with size NxM where N is the number of channels, and M is the number of samples.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def __call__(self, signal, *args, **kwargs):
        result = signal
        result = self.parent_preprocessor(result) if self.parent_preprocessor is not None else result
        return self.process_signal(result)