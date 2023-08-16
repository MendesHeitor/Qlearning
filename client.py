from q_learning import q_learning, aux
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ======================================== Constants =============================================
    ROWS = 96
    COLS = 3

    ACTIONS = ["left", "right", "jump"]
    DIRECTIONS = {"NORTH":0, "EAST":1, "SOUTH":2, "WEST":3}

    #Initializing the Q matrix -> 96 rows and 3 columns
    Q_MATRIX = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    # =========== Q-Learning Trainer ===========
    ql = q_learning(ROWS, COLS, Q_MATRIX, ACTIONS, DIRECTIONS)
      

    # =========== Training Setup ===========

    # Initial state -> need to be equal to the initial state of the game -> must check
    platform = 0
    direction = DIRECTIONS["NORTH"]
    
    # =========== Training ===========
    # aux.clear_table_txt("resultado.txt", ROWS, COLS)
    ql.train_one_source(port=2037, epochs=3000, initial_plat=platform, initial_dir=direction, alpha=0.6, gamma=0.75, epsilon=0.15)

    # =========== Trying to reach the goal using the built Q matrix ===========
    aux.q_table_runner(filename="resultado.txt", port=2037, initial_plat=platform, initial_dir=direction)

if  __name__ == "__main__":
    main()