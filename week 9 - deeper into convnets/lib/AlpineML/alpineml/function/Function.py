from abc import ABC, abstractmethod


class Function(ABC):
    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def derivative(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.__class__.apply(*args, **kwargs)
