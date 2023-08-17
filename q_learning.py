import random as r
import connection as cn

class q_learning:

    def __init__(self, rows, cols, actions, directions, filename):
        self.ROWS = rows
        self.COLS = cols
        self.Q_MATRIX = aux.read_q_matrix(filename)
        self.ACTIONS = actions
        self.DIRECTIONS = directions

    def q_update(self, state:bin, action:int, next_state:bin, reward:int, alpha:float, gamma:float) -> float:
        """
        Update the Q matrix: Uses bellman equation
        
        alpha -> learning rate
        gamma -> discount factor
        """
        estimate_q = reward + gamma * max(self.Q_MATRIX[next_state])
        q_value = self.Q_MATRIX[state][action] + alpha * (estimate_q - self.Q_MATRIX[state][action])

        return q_value

    def q_table_runner(self, port:int, initial_plat:int, initial_dir:str) -> None:
        """
        Run the table
        """
        won = False

        s = cn.connect(port)

        platform = initial_plat 
        direction = initial_dir
        state = aux.state_pack(platform, direction)

        terminal_state = False

        while not terminal_state:
            action = aux.choose_action(-1, state, self.Q_MATRIX)
            next_state, reward_next = cn.get_state_reward(s, self.ACTIONS[action])
            state = next_state

            # Check for terminal state
            if reward_next == 300:
                terminal_state = True
                won = True
            elif reward_next == -100:
                terminal_state = True
        
        print(f"Game won: {won}")

    
    def train_one_source(self, port:int, initial_plat:int, initial_dir:int,  epochs:int, alpha:float, gamma:float, epsilon:float) -> None:
        """
        Train the Q matrix for one source
            parameters:
                initial_plat -> initial platform
                initial_dir -> initial direction
                epochs -> number of epochs
                alpha -> learning rate
                gamma -> discount factor
        """
        back_to_back_wins = 0

        wins = 0
        deaths = 0

        # Connection with unity game
        s = cn.connect(port)

        # Initial state -> need to be equal to the initial state of the game -> must check
        platform = initial_plat 
        direction = initial_dir
        state = aux.state_pack(platform, direction)

        # ========= Training =========

        try:
            for i in range(epochs):

                terminal_state = False
                
                if(back_to_back_wins == 10):
                    break

                while (not terminal_state):
                    action = aux.choose_action(epsilon, state, self.Q_MATRIX)
                    next_state, reward_next = cn.get_state_reward(s, self.ACTIONS[action])

                    self.Q_MATRIX[aux.convert_to_int(state)][action] = self.q_update(aux.convert_to_int(state), action, aux.convert_to_int(next_state), reward_next, alpha, gamma)
                    
                    state = next_state

                    # Check for terminal state
                    if reward_next == 300:
                        terminal_state = True
                        wins += 1
                        back_to_back_wins += 1
                    elif reward_next == -100:
                        terminal_state = True
                        deaths += 1
                        back_to_back_wins = 0

                print(f"Epoch: {i} complete [=========================================]")
                print(f"Sucess rate: {100 * (wins/(wins+deaths))}%")
                
        except KeyboardInterrupt:
            aux.write_q_matrix("resultado.txt", self.Q_MATRIX)
                


        aux.write_q_matrix("resultado.txt", self.Q_MATRIX)
        aux.print_table_actions(self.ACTIONS, self.Q_MATRIX)


# ======================================== Helper Functions =============================================

class aux:
    @classmethod
    def state_pack(self, platform: int, direction: int) -> str:
        """
        Pack the state into a string
        """
        return '0b' + format(platform, '05b') + format(direction, '02b')

    @classmethod
    def convert_to_int(self, state: str) -> int:
        """
        Convert the state string to int
        """
        return int(state[2:], 2)
    
    @classmethod
    def read_q_matrix(self, filename:str) -> [[float]]:
        """
        Read the Q matrix from a txt file
        """
        q_matrix = []
        with open(filename, 'r') as file:
            for line in file:
                row = list(map(float, line.strip().split()))
                q_matrix.append(row)
        return q_matrix
    
    @classmethod
    def write_q_matrix(self, filename:str, q_matrix:[[float]]) -> None:
        """
        Write the Q matrix to a txt file
        """
        with open(filename, 'w') as file:
            for row in q_matrix:
                file.write(" ".join(map(str, row)) + '\n')
    

    @classmethod
    def print_table_actions(self, actions: [str], q_table: [[float]]) -> None:
        """
        Evaluate the table and print the best action for each state
        """
        for state in range(len(q_table)):
            action  = actions[q_table[state].index(max(q_table[state]))]

            if all(val == 0 for val in q_table[state]):
                action = "None"
            print(f"State: {state} -> Action: {action}")

    @classmethod
    def clear_table_txt(self, filename:str, rows:int, cols:int) -> None:
        """
        Clear the table
        """
        row = [0.000000 for _ in range(cols)]
        with open(filename, 'w') as file:
            for _ in range(rows):
                file.write(" ".join(map(str, row)) + '\n')
    
    @classmethod
    def choose_action(self, epsilon:float, state:bin, q_matrix:[[float]]) -> int:
        """
        Chooses a actions based on eploitation and exploration

        0 -> left
        1 -> right
        2 -> jump
        """        
        if r.random() < epsilon:
            return r.randint(0, 2)

        return q_matrix[aux.convert_to_int(state)].index(max(q_matrix[aux.convert_to_int(state)]))



        
        
        