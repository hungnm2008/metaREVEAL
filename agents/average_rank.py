from agents.base_agent import *
import copy
import operator
import numpy as np

class Average_Rank(Base_Agent):
    """
    Model the Average_Rank agent
    """
    def __init__(self, agent_name, env):
        super().__init__(agent_name, env)
        self.agent_name = agent_name
        self.all_episodes_rank = []
        self.current_episode_scores = []

    def precompute_rank(self):
        """
        Compute average rank
        """
        for i in range(self.env.nA):
            action = i
            clone_env = copy.deepcopy(self.env)
            done = False
            ep_reward = 0
            while not done:
                next_state, reward, done, info = clone_env.step(action)
                ep_reward = ep_reward + reward.item()
            self.current_episode_scores.append(ep_reward)

        self.current_episode_rank = [operator.itemgetter(0)(t) for t in sorted(enumerate(self.current_episode_scores,1), key=operator.itemgetter(1), reverse=True)]
        self.all_episodes_rank.append(self.current_episode_rank)

        # Clear current episode
        self.current_episode_scores = []
        self.current_episode_rank = []

    def select_action(self, state, evaluate):
        """
          Return the next action
        """
        a = np.array(self.all_episodes_rank)
        a = a.mean(axis=0)
        action = np.argmin(a)

        return torch.tensor(action)
