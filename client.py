from q_learning import q_learning, aux

def main():
    # ======================================== Constants =============================================
    ROWS = 96
    COLS = 3

    ACTIONS = ["left", "jump", "right"]
    DIRECTIONS = {"NORTH":0, "EAST":1, "SOUTH":2, "WEST":3}

    #Initializing the Q matrix -> 96 rows and 3 columns
    Q_MATRIX = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    # Reward array for each platform -> used only in the initialization of the training
    REWARDS = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -10, -8, -6, -12, -7, -5, -4, -3, -2, -1]

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

    # =========== Q-Learning Object ===========
    ql = q_learning(ROWS, COLS, Q_MATRIX, ACTIONS, DIRECTIONS, REWARDS, BEST_ACTIONS)
    
    # =========== Training Setup ===========

    # Initial state -> need to be equal to the initial state of the game -> must check
    platform = 0
    direction = DIRECTIONS["NORTH"]
    
    ql.train_one_source(port=2037, epochs=50, initial_plat=platform, initial_dir=direction, alpha=0.12, gamma=0.75)

    Q_MATRIX = aux.read_q_matrix("resultado.txt")
    aux.evaluate_table(BEST_ACTIONS, Q_MATRIX)

if  __name__ == "__main__":
    main()