from abc import ABC, abstractmethod
from copy import copy


class Function(ABC):
    def __init__(self):
        self.is_derivative = False

    @property
    def derivative(self):
        my_copy = copy(self)
        my_copy.is_derivative = True
        return my_copy

    @abstractmethod
    def apply(self, *args):
        pass

    @abstractmethod
    def apply_derivative(self, *args):
        pass

    def __call__(self, *args):
        if self.is_derivative:
            return self.apply_derivative(*args)
        else:
            return self.apply(*args)
