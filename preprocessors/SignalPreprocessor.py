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