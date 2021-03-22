

import random

class TestAgent:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, obs, info):
        x = random.random()
        action = {}
        if x < 0.3:
            action["type"] = 0    #forward
        elif x < 0.6:
            action["type"] = 1    #turn left
        else:
            action["type"] = 2    #turn right
        #action = np.random.uniform(low=-1, high=1, size=(ACTION_DIM,))
        return action