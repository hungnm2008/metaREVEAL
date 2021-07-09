class Baseline_Average_Rank(Base_Agent):
    """
    Baseline for Learning curve game
    """
    def __init__(self, agent_name, env):
        super().__init__(agent_name, env)
        self.agent_name = agent_name
        self.all_episodes_rank = []
        self.current_episode_scores = []

    def precompute_rank(self):
        for i in range(self.env.nA):
            action = i
#             current_algo_name = self.env.get_algo_name_by_index(i)

            clone_env = copy.deepcopy(self.env)
            done = False
            ep_reward = 0
            while not done:
                next_state, reward, done, info = clone_env.step(action)
                ep_reward = ep_reward + reward.item()
#             max_score = 0.0
#             while amount_of_time < self.env.max_number_of_steps:
#                 score = self.env.current_dataset[current_algo_name].get_value(amount_of_time)
#                 if score > max_score:
#                     max_score = score
#                 if self.env.learning_type == "fixed_time_learning":
#                     reward =
#                 elif self.env.learning_type == "fixed_time_learning":
#                     reward =
#                 amount_of_time = amount_of_time + self.env.time_for_each_action


            self.current_episode_scores.append(ep_reward)

        self.current_episode_rank = [operator.itemgetter(0)(t) for t in sorted(enumerate(self.current_episode_scores,1), key=operator.itemgetter(1), reverse=True)]
        self.all_episodes_rank.append(self.current_episode_rank)

        # Clear current episode
        self.current_episode_scores = []
        self.current_episode_rank = []

    def select_action(self, state, evaluate):
        """
          Return an random action
          :param state: the current state
          :return action: next action to take
        """
        a = np.array(self.all_episodes_rank)
        a = a.mean(axis=0)
        action = np.argmin(a)
#         print("action=", action)

        return torch.tensor(action)
