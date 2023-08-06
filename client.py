import connection as cn
import random as r

s = cn.connect(2037)

ROWS = 96
COLS = 3

ACTIONS = ["left", "jump", "right"]
DIRECTIONS = {"NORTH":00, "EAST":1, "SOUTH":2, "WEST":3}

Q_MATRIX = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Reward array for each platform -> used only in the initialization of the training
REWARDS = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -10, -8, -6, -12, -7, -5, -4, -3, -2, -1]

def state_unpack(state: str) -> (int, int):
    """
    Unpack the state into a tuple containing platform and direction
    """
    platform = int(state[2:7], 2)
    direction = int(state[7:], 2)

    return platform, direction

def state_pack(platform: int, direction: int) -> str:
    """
    Pack the state into a string
    """
    return '0b' + format(platform, '05b') + format(direction, '02b')

def convert_to_int(state: str) -> str:
    """
    Convert the state string to int
    """
    return int(state[2:], 2)

def choose_action() -> int:
    """
    Choose a random action to take:

    1 - turn left
    2 - jump
    3 - turn right
    """
    random_n = r.randint(0, 2)
    return random_n

def check_terminal(reward:int) -> bool:
    """
    Check if the game is in a terminal state
    """
    if reward == -100 or reward == 300:
        return True
    else:
        return False

def q_update(state:bin, action:int, next_state:bin, reward:int, alpha:float, gamma:float) -> float:
    """
    Update the Q matrix: Uses bellman equation
    
    alpha -> learning rate
    gamma -> discount factor
    """
    estimate_q = reward + gamma * max(Q_MATRIX[next_state])
    q_value = Q_MATRIX[state][action] + alpha * (estimate_q - Q_MATRIX[state][action])

    return q_value

def write_q_matrix(filename:str) -> None:
    """
    Write the Q matrix to a txt file
    """
    with open(filename, 'w') as file:
        for row in Q_MATRIX:
            file.write(" ".join(map(str, row)) + '\n')

def read_q_matrix(filename:str) -> list:
    """
    Read the Q matrix from a txt file
    """
    q_matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            q_matrix.append(row)
    return q_matrix

def print_table():
    """
    Print the Q matrix
    """
    for i in range(ROWS):
        print(Q_MATRIX[i])

# Prepare Training

# Initial state -> need to be equal to the initial state of the game -> must check
platform = 0
direction = DIRECTIONS["NORTH"]
state = state_pack(platform, direction)

# Initial reward
reward = REWARDS[platform] 

# Hyperparameters: learning rate and discount factor, respectively
alpha = 0.15
gamma = 0.7

Q_MATRIX = read_q_matrix("resultado.txt")

# Training Loop
for i in range(1000):

    terminal_state = False

    while(not terminal_state):
        action = choose_action()
        next_state, reward_next = cn.get_state_reward(s, ACTIONS[action])

        Q_MATRIX[convert_to_int(state)][action] = q_update(convert_to_int(state), action, convert_to_int(next_state), reward, alpha, gamma)
        
        reward = reward_next
        state = next_state

        terminal_state = check_terminal(reward)
    
    print(f"=============================== Epoch:  {i}   ======================================")
    print_table()

write_q_matrix("resultado.txt")