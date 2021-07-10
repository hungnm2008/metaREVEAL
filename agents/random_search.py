from agents.base_agent import *
import random
import time
class Random_Agent(Base_Agent):
    def __init__(self, agent_name, env):
        super().__init__(agent_name, env)

    def select_action(self, state, evaluate=False):
        """
          Return a random action
        """
        random_action = torch.tensor(random.randint(0,self.env.nA-1))
        return random_action
