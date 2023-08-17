from q_learning import q_learning, aux
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

def main():
    # ======================================== Constants =============================================
    ROWS = 96
    COLS = 3

    ACTIONS = ["left", "right", "jump"]
    DIRECTIONS = {"NORTH":0, "EAST":1, "SOUTH":2, "WEST":3}
    FILENAME = "resultado.txt"

    # =========== Q-Learning Trainer ===========
    ql = q_learning(ROWS, COLS, ACTIONS, DIRECTIONS, FILENAME)
      

    # =========== Training Setup ===========

    # Initial state -> need to be equal to the initial state of the game -> must check
    platform = 0
    direction = DIRECTIONS["NORTH"]
    
    # =========== Training ===========
    # aux.clear_table_txt("resultado.txt", ROWS, COLS)
    # ql.train_one_source(port=2037, epochs=30, initial_plat=platform, initial_dir=direction, alpha=0.65, gamma=0.8, epsilon=0.4)

    # =========== Trying to reach the goal using the built Q matrix ===========
    for platform in range(2, 3):
        # platform = int(input("Insira a plataforma de inicio: "))

        print(f"Altere a plataforma para {platform} e reinicie o jogo (mate o amongois)")
        input("Pressione Enter para começar")
        
        ql.q_table_runner(port=2037, initial_plat=platform, initial_dir=direction)

if  __name__ == "__main__":
    main()