import random as r
import connection as cn

class q_learning:
    def __init__(self, rows, cols, q_matrix, actions, directions, rewards, best_actions):
        self.ROWS = rows
        self.COLS = cols
        self.Q_MATRIX = q_matrix
        self.ACTIONS = actions
        self.DIRECTIONS = directions
        self.REWARDS = rewards
        self.BEST_ACTIONS = best_actions

    def q_update(self, state:bin, action:int, next_state:bin, reward:int, alpha:float, gamma:float) -> float:
        """
        Update the Q matrix: Uses bellman equation
        
        alpha -> learning rate
        gamma -> discount factor
        """
        estimate_q = reward + gamma * max(self.Q_MATRIX[next_state])
        q_value = self.Q_MATRIX[state][action] + alpha * (estimate_q - self.Q_MATRIX[state][action])

        return q_value

    def write_q_matrix(self, filename:str) -> None:
        """
        Write the Q matrix to a txt file
        """
        with open(filename, 'w') as file:
            for row in self.Q_MATRIX:
                file.write(" ".join(map(str, row)) + '\n')

    def print_table(self) -> None:
        """
        Print the Q matrix
        """
        for i in range(self.ROWS):
            print(self.Q_MATRIX[i])
    
    def train_one_source(self, port:int, initial_plat:int, initial_dir:int,  epochs:int, alpha:float, gamma:float) -> None:
        """
        Train the Q matrix for one source
            parameters:
                initial_plat -> initial platform
                initial_dir -> initial direction
                epochs -> number of epochs
                alpha -> learning rate
                gamma -> discount factor
        """

        # Connection with unity game
        s = cn.connect(port)

        # Initial state -> need to be equal to the initial state of the game -> must check
        platform = initial_plat 
        direction = initial_dir
        state = aux.state_pack(platform, direction)

        # Initial reward
        reward = self.REWARDS[platform] 

        self.Q_MATRIX = aux.read_q_matrix("resultado.txt")

        # ========= Training =========

        for i in range(epochs):
            terminal_state = False

            while(not terminal_state):
                action = aux.choose_action()
                next_state, reward_next = cn.get_state_reward(s, self.ACTIONS[action])

                self.Q_MATRIX[aux.convert_to_int(state)][action] = self.q_update(aux.convert_to_int(state), action, aux.convert_to_int(next_state), reward, alpha, gamma)
                
                reward = reward_next
                state = next_state

                terminal_state = aux.check_terminal(reward)
            
            print(f"Epoch: {i} complete [=========================================]")
            print(f"Acuracy: {aux.evaluate_table(self.BEST_ACTIONS, self.Q_MATRIX)}%")

        aux.write_q_matrix("resultado.txt")

# ======================================== Helper Functions =============================================

class aux:
    @classmethod
    def state_unpack(self, state: str) -> (int, int):
            """
            Unpack the state into a tuple containing platform and direction
            """
            platform = int(state[2:7], 2)
            direction = int(state[7:], 2)

            return platform, direction

    @classmethod
    def state_pack(self, platform: int, direction: int) -> str:
        """
        Pack the state into a string
        """
        return '0b' + format(platform, '05b') + format(direction, '02b')

    @classmethod
    def convert_to_int(self, state: str) -> str:
        """
        Convert the state string to int
        """
        return int(state[2:], 2)

    @classmethod
    def choose_action(self) -> int:
        """
        Choose a random action to take:

        1 - turn left
        2 - jump
        3 - turn right
        """
        random_n = r.randint(0, 2)
        return random_n

    @classmethod
    def check_terminal(self, reward:int) -> bool:
        """
        Check if the game is in a terminal state
        """
        if reward == -100 or reward == 300:
            return True
        else:
            return False
    
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
    def print_table_actions(self, actions: [str], q_table: [[float]]) -> None:
        """
        Evaluate the table and print the best action for each state
        """
        for state in range(len(q_table)):
            print(f"State: {state} -> Action: {actions[q_table[state].index(max(q_table[state]))]}")
    
    @classmethod
    def evaluate_table(self, best_actions:[[int]], q_table:[[float]]) -> float:
        """
        Evaluate the table based on a ideal table, returns the accuracy (%) of the table
        """
        correct = 0
        for state in range(len(q_table)):
            if q_table[state].index(max(q_table[state])) in best_actions[state]:
                correct += 1
        return correct / len(q_table) * 100
        