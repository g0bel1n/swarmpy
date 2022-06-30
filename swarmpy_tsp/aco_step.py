from abc import ABC, abstractmethod


class ACO_Step(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self) -> dict:
        pass

    def get_run_args(self):
        return self.run.__code__.co_varnames[:self.run.__code__.co_argcount]
