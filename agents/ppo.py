from agents.base_agent import *
from nn_builder.pytorch.CNN import CNN
from helpers import *
from torch.distributions import Categorical
import copy
import sys
class PPO_Discrete(Base_Agent):
    """
    Proximal Policy Optimization agent for discrete action space
    This code is adapted from: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
    """
    def __init__(self, agent_name, env, gamma, batch_size, epsilon, epsilon_decay_rate):
        super().__init__(agent_name, env)
        self.gamma = gamma # The weight for future rewards
        self.batch_size = batch_size
        self.buffer = Replay()
        self.L1loss = torch.nn.SmoothL1Loss()
        self.mse = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_output_size = self.env.nA

        self.action_types = "DISCRETE"
        self.learning_iterations_per_round = 1
        self.episodes_per_learning_round = 1
        self.clip_epsilon = 0.2
        self.gradient_clipping_norm = 5

        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        self.policy_new = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
                            output_activation="softmax",
#                             dropout=0.5,
                            initialiser="xavier",
                            batch_norm=False)
        self.policy_old = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
                            output_activation="softmax",
#                             dropout=0.5,
                            initialiser="xavier",
                            batch_norm=False)

        self.policy_new.to(self.device)
        self.policy_old.to(self.device)

        self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))

        self.policy_new_optimizer = torch.optim.Adam(self.policy_new.parameters(), lr=1e-4, eps=1e-4)

        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []

    def train(self):
        """
        Train PPO agent
        """
        loss = 0.0
        loss = self.policy_learn()
        self.copy_policy()
        self.many_episode_states, self.many_episode_actions, self.many_episode_rewards = [], [], []
        return loss.detach().item()


    def select_action(self, state, evaluate):
        """
        Output the next action
        """
        state = [state]
        state = torch.stack(state)
        state = state.to(self.device, dtype=torch.float)
        action_probabilities = self.policy_new(state)
        action_distribution = Categorical(action_probabilities)

        if not evaluate:
            action = action_distribution.sample().cpu()
        else:
            action = torch.argmax(action_probabilities).item()

        if evaluate:
            if self.env.freely_moving:
                self.list_evaluation_values.append(action_probabilities.reshape(self.env.number_of_rows, self.env.number_of_columns))
            else:
                self.list_evaluation_values.append(action_probabilities.reshape(1, self.env.nA))

        return torch.tensor(action)

    def policy_learn(self):
        """
        A learning iteration for the policy
        """
        # Calculated all discounted returns
        all_discounted_returns = self.calculate_all_discounted_returns()

        loss = 0.0

        for _ in range(self.learning_iterations_per_round):
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)
        return loss

    def calculate_all_discounted_returns(self):
        """
        Calculates the cumulative discounted return for each episode which we will then use in a learning iteration
        """
        all_discounted_returns = []
        for episode in range(len(self.many_episode_states)):
            discounted_returns = [0]
            for ix in range(len(self.many_episode_states[episode])):
                return_value = self.many_episode_rewards[episode][-(ix + 1)] + self.gamma*discounted_returns[-1]
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns

    def calculate_all_ratio_of_policy_probabilities(self):
        """
        For each action calculates the ratio of the probability that the new policy would have picked the action vs.
         the probability the old policy would have picked it. This will then be used to inform the loss
        """
        if torch.cuda.is_available():
            all_states = [state.cuda() for states in self.many_episode_states for state in states]
        else:
            all_states = [state for states in self.many_episode_states for state in states]
        all_actions = [[action] if self.action_types == "DISCRETE" else action for actions in self.many_episode_actions for action in actions ]
        all_states = torch.stack([states for states in all_states])

        all_actions = torch.stack([torch.Tensor(actions).float().to(self.device) for actions in all_actions])
        all_actions = all_actions.view(-1, len(all_states))

        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_new, all_states, all_actions)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_old, all_states, all_actions)
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """
        Calculates the log probability of an action occuring given a policy and starting state
        """
        policy_output = policy.forward(states).to(self.device)
#             policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)

        policy_distribution = Categorical(policy_output)  # this creates a distribution to sample from

        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):
        """
        Calculates the PPO loss
        """
        all_ratio_of_policy_probabilities = torch.squeeze(torch.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities,
                                                        min = -sys.maxsize,
                                                        max = sys.maxsize)
        all_discounted_returns = torch.tensor(all_discounted_returns).to(all_ratio_of_policy_probabilities)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * self.clamp_probability_ratio(all_ratio_of_policy_probabilities)
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        loss = -torch.mean(loss)
        return loss

    def clamp_probability_ratio(self, value):
        """
        Clamps a value between a certain range determined by hyperparameter clip epsilon
        """
        return torch.clamp(input=value, min=1.0 - self.clip_epsilon,
                                  max=1.0 + self.clip_epsilon)

    def take_policy_new_optimisation_step(self, loss):
        """
        Takes an optimisation step for the new policy
        """
        self.policy_new_optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), self.gradient_clipping_norm)  # clip gradients to help stabilise training
        self.policy_new_optimizer.step()  # this applies the gradients

    def copy_policy(self):
        """
        Sets the old policy's parameters equal to the new policy's parameters
        """
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)

    def save_result(self):
        """
        Save the results seen by the agent in the most recent experiences
        """
        for ep in range(len(self.many_episode_rewards)):
            total_reward = np.sum(self.many_episode_rewards[ep])
            self.game_full_episode_scores.append(total_reward)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()
