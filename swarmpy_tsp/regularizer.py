from aco_step import ACO_Step
import numpy as np

class Regularizer(ACO_Step):

    def __init__(self) -> None:
        super().__init__()
    
    def run(self, G : dict[str, np.ndarray]) -> dict:
        
        return super().run()