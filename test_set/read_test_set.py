import numpy as np

def read_test_set(filename:str):
    position = []
    with open(f'{filename}.txt') as f:
        position.extend(line.split()[1:] for line in f.readlines())
    position = np.array(position, dtype=float)

    with open(f'{filename}_solution.txt') as f:
        solution=[int(line) for line in f.readlines()]

        
    return solution, position

