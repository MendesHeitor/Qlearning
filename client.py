from q_learning import q_learning, aux
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ======================================== Constants =============================================
    ROWS = 96
    COLS = 3

    ACTIONS = ["left", "jump", "right"]
    DIRECTIONS = {"NORTH":0, "EAST":1, "SOUTH":2, "WEST":3}

    #Initializing the Q matrix -> 96 rows and 3 columns
    Q_MATRIX = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    BEST_ACTIONS = [
        [1], [2], [0, 2], [0],
        [0], [1], [2], [0, 2],
        [0, 2], [0], [1], [2],
        [0, 2], [0], [1], [2],
        [0], [1], [1], [2],
        [0, 1], [1], [2], [0, 2],
        [0], [1], [2], [0, 2],
        [1], [1, 2], [0, 2], [0],
        [0], [1], [2], [0, 2],
        [0], [1], [2], [0, 2],
        [0], [1], [2], [0, 2],
        [0, 2], [0], [1], [2],
        [0, 2], [0], [1], [2],
        [0], [1], [2], [0, 2],
        [1], [2], [0, 2], [0],
        [1], [2], [0, 2], [0],
        [1], [2], [0, 2], [0],
        [0], [1], [2], [0, 2],
        [0], [1], [2], [0, 2],
        [0], [1], [2], [0, 2],
        [0], [1], [2], [0, 2],
        [1], [2], [0, 2], [0],
        [1], [2], [0, 2], [0],
        [0], [1], [2], [0, 2]
    ]

    # =========== Q-Learning Trainer ===========
    ql = q_learning(ROWS, COLS, Q_MATRIX, ACTIONS, DIRECTIONS, BEST_ACTIONS)
      

    # =========== Training Setup ===========

    # Initial state -> need to be equal to the initial state of the game -> must check
    platform = 17
    direction = DIRECTIONS["NORTH"]
    
    # =========== Training ===========
    # aux.clear_table_txt("resultado.txt", ROWS, COLS) 
    ql.train_one_source(port=2037, epochs=100, initial_plat=platform, initial_dir=direction, alpha=0.45, gamma=0.9, epsilon=0.08)

    
    Q_MATRIX = aux.read_q_matrix("resultado.txt")
    print(f"Final accuracy: {aux.evaluate_table(BEST_ACTIONS, Q_MATRIX)}")

if  __name__ == "__main__":
    main()