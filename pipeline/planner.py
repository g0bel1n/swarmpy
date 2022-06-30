from typing import Optional
import numpy as np
from .aco_step import ACO_Step


class Planner(ACO_Step):
    def __init__(self, ant_params: dict[str, float], how="simple"):
        self.ant_params = ant_params

    def run(self, nb_iter: int):
        return {'ant_params': self.ant_params}

