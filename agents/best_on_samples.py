from agents.base_agent import *
import math
import numpy as np
class Best_On_Samples(Base_Agent):
    """
    Model the Best On Samples agent
    """
    def __init__(self, agent_name, env, samples_time_for_each_algo):
        super().__init__(agent_name, env)
        self.agent_name = agent_name
        self.samples_time_for_each_algo = samples_time_for_each_algo
        self.total_number_of_trials = math.ceil(self.samples_time_for_each_algo/self.env.time_for_each_action)

    def init(self):
        a = [i for i in range(self.env.nA)]
        self.list_precomputed_actions = a*self.total_number_of_trials
        self.current_max_scores = np.zeros(self.env.nA)
        self.current_times_spent = np.zeros(self.env.nA)

    def select_action(self, state, evaluate):
        """
          Return the next action
        """
        if len(self.list_precomputed_actions)>0:
            action = self.list_precomputed_actions.pop(0)

            if self.env.dataset == "artificial":
                current_algo_name = self.env.get_algo_name_by_index_artificial(action)
            else:
                current_algo_name = self.env.get_algo_name_by_index(action)

            self.current_times_spent[action] += self.env.time_for_each_action
            score = self.env.current_dataset[current_algo_name].get_value(self.current_times_spent[action])
            self.current_max_scores[action] = max(score, self.current_max_scores[action])

        else:
            action = np.argmax(self.current_max_scores)

        return torch.tensor(action)
