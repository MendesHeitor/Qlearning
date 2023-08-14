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
    platform = 0
    direction = DIRECTIONS["NORTH"]
    
    # =========== Training ===========
    # aux.clear_table_txt("resultado.txt", ROWS, COLS) 
    ql.train_one_source(port=2037, epochs=20, initial_plat=platform, initial_dir=direction, alpha=0.75, gamma=0.7, epsilon=0.1)

    # =========== Varying Parameters ===========

    # alpha_values = np.arange(0.05, 1.0, 0.5)
    # gamma_values = np.arange(0.05, 1.0, 0.5)
    # epsilon_values = np.arange(0.05, 1.0, 0.5)

    # accuracy_values = []

    # for a in alpha_values:
    #     for g in gamma_values:
    #         for e in epsilon_values:
    #             aux.clear_table_txt("resultado.txt", ROWS, COLS) 
    #             ql.train_one_source(port=2037, epochs=20, initial_plat=platform, initial_dir=direction, alpha=a, gamma=g, epsilon=e)
    #             accuracy_values.append(aux.evaluate_table(BEST_ACTIONS, Q_MATRIX))

    # # Reshape accuracy values into a grid for plotting
    # accuracy_grid = np.array(accuracy_values).reshape(len(alpha_values), len(gamma_values), len(epsilon_values))

    # # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(alpha_values, gamma_values)
    # ax.plot_surface(X, Y, accuracy_grid[:,:,0])  # Plotting for the first epsilon value

    # ax.set_xlabel('Alpha')
    # ax.set_ylabel('Gamma')
    # ax.set_zlabel('Accuracy')
    
    Q_MATRIX = aux.read_q_matrix("resultado.txt")
    print(f"Final accuracy: {aux.evaluate_table(BEST_ACTIONS, Q_MATRIX)}")

if  __name__ == "__main__":
    main()