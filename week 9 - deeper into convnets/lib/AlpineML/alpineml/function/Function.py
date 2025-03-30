from abc import ABC, abstractmethod


class Function(ABC):
    @staticmethod
    @abstractmethod
    def apply(*args):
        pass

    @staticmethod
    @abstractmethod
    def derivative(*args):
        pass

    def __call__(self, *args):
        return self.__class__.apply(*args)
