import connection as cn
import random as r

s = cn.connect(2037)

state, reward = cn.get_state_reward(s, "jump")

def state_unpack(state: str) -> tuple(int, int):
    """
    Unpack the state into a tuple containing platform and direction
    """
    position = int(state[:5], 2)
    direction = int(state[5:], 2)

    return position, direction

def choose_action() -> int:
    """
    Choose a random action to take:

    1 - turn left
    2 - jump
    3 - turn right
    """
    random_n = r.randint(1, 3)
    return random_n

def get_next_state(action) -> tuple(int, int, int):
    """
    get the next state and reward from the game
    """
    next_state, reward = cn.get_state_reward(s, action)
    position, direction = state_unpack(next_state)

    return position, direction, reward

# def q_update(state, )


print (f"reward: {reward}, state: {state}")

