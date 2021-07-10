import torch
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import copy
import random
import numpy as np
import math
class Replay():
    """
    Memory for storing experience
    """
    def __init__(self):
        self.buffer = deque(maxlen=10000)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer
        :param state: The state before taking the action
        :param action: action taken
        :param reward: Reward for taking that action
        :param next_state: The state that the agent enters after taking the action
        :param done: Whether the agent finishes the game
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
          Return a batch of samples from the experience buffer
          :param batch_size: The number of sample that you want to take
          :return the batch of samples but decomposed into lists of states, actions, rewards, next_states
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # Random samples
        samples = random.sample(self.buffer, batch_size)

        for s in samples:
            state = s[0]
            action = s[1]
            reward = s[2]
            next_state = s[3]
            done = s[4]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

class Base_Agent():
    """
    A base for all agents
    """
    def __init__(self, agent_name, env):
        """ Initialization
        :param agent_name: (str) name of the agent, e.g., SAC_agent
        :param env: (object) the environment where agents are running in
        """
        self.agent_name = agent_name
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.epsilon = None
        self.epsilon_decay_rate = None

        self.fig = plt.figure(constrained_layout=True, figsize = (20, 20))
        widths = [3, 0.05]   # [0.02, 3, 1, 0.02]
        heights = [6, 1]
        spec = self.fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights, wspace=0.2, hspace=0.3)

        self.ax = self.fig.add_subplot(spec[0, :])
        self.ax3 = self.fig.add_subplot(spec[1,0])
        self.cbar_ax3 = self.fig.add_subplot(spec[1,1])

        self.fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    def select_action(self, state, evaluate):
        """ Select the next action
        :param state: (tensor) current state
        :param evaluate: (boolean) whether agents are running in an evaluating episode
        :return next_action: the next action to take
        """
        pass

    def train():
        """ Train neural networks
        """
        pass

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """
        Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def my_func(self, i):
        """
        For plotting animations
        """
        if self.env.env_name == "segment":
            if len(self.list_evaluation_states) > 0: # if not empty
                self.ax.cla()
                plt1 = sns.heatmap(self.list_evaluation_states[i, ...],
                            ax = self.ax,
                            cbar = True,
                            cbar_ax = self.cbar_ax,
                            vmin = self.list_evaluation_states.min(),
                            vmax = self.list_evaluation_states.max(),
                            cmap="Blues",
                            annot=True)
                self.ax.set_title(self.list_evaluation_captions[i])
                c_bar = plt1.collections[0].colorbar
                c_bar.set_ticks([-1, 0, 1])
                c_bar.set_ticklabels(['Unknown', 'White', 'Black'])

        elif self.env.env_name == "battleship":
            if len(self.list_ground_truth) > 0: # if not empty
                self.ax.cla()
                plt1 = sns.heatmap(self.list_ground_truth[i, ...],
                            ax = self.ax,
                            cbar = True,
                            cbar_ax = self.cbar_ax,
                            vmin = self.list_ground_truth.min(),
                            vmax = self.list_ground_truth.max(),
                            cmap="Blues",
                            annot=True)
                self.ax.set_title("Ground truth")

            if len(self.list_evaluation_states) > 0: # if not empty
                self.ax2.cla()
                plt2 = sns.heatmap(self.list_evaluation_states[i, ...],
                            ax = self.ax2,
                            cbar = True,
                            cbar_ax = self.cbar_ax2,
                            vmin = self.list_evaluation_states.min(),
                            vmax = self.list_evaluation_states.max(),
                            cmap="Blues",
                            annot=True)
                self.ax2.set_title(self.list_evaluation_captions[i])
                c_bar = plt2.collections[0].colorbar
                c_bar.set_ticks([-1])
                c_bar.set_ticklabels(['Unknown'])

        elif self.env.env_name == "learning_curve":
            if len(self.list_loss_evaluation) > 0: # if not empty
                self.ax.cla()
                self.list_loss_evaluation = self.list_loss_evaluation.reset_index(drop=True)
                self.list_loss_evaluation.replace(to_replace=[-1],value=np.NaN,inplace=True)

                df = self.list_loss_evaluation.head(i+1)
                palette = sns.color_palette('Spectral',self.env.nA)
                df = df.melt('time_step', var_name='Algorithm',  value_name='loss')
                plt1 = sns.lineplot(x="time_step", y="loss", data=df, hue='Algorithm', palette=palette, ax=self.ax, marker="o")

                plt1.set_xlabel('time spent by the agent for each algorithm (seconds)')
                plt1.set_ylabel('nauc score')

                self.ax.set_xlim([0, int(self.env.max_number_of_steps/10)])
                self.ax.set_xticks(range(0, int(self.env.max_number_of_steps/10), int(self.env.max_number_of_steps/50)))
                self.ax.set_ylim([0, 1.05])
                self.ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

                self.ax.set_title("Dataset: " + self.env.current_dataset_name) # "  Best algo: " + str(self.env.best_algo)) #str(self.env.best_algo_time) + " hours for Algorithm "

                plt1.fill_between(self.upperbound['time_step'], self.upperbound[0], self.lowerbound[0], facecolor='dimgray', alpha=0.1, label='area between upperbound and lowerbound')

    def visualize(self, download):
        """
        Visualize animations
        """

        if len(self.list_evaluation_values)>0: # if not empty
            self.list_evaluation_values = torch.stack(self.list_evaluation_values).cpu()
        if len(self.list_ground_truth)>0: # if not empty
            self.list_ground_truth = torch.stack(self.list_ground_truth).cpu()

        self.upperbound = self.list_loss_evaluation.groupby(['time_step']).max()
        self.upperbound = self.upperbound.max(axis = 1).reset_index()

        self.lowerbound = self.list_loss_evaluation.groupby(['time_step']).min()
        self.lowerbound = self.lowerbound.min(axis = 1).reset_index()

        anim = FuncAnimation(fig = self.fig, func = self.my_func, frames = len(self.list_loss_evaluation), interval = 2000, blit = False)

        if download:
            filename = "evaluation_"
            writer = PillowWriter(fps=0.8)
            date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            filename = filename + self.env.env_name + "_" + self.agent_name + "_" + date + ".gif"
            anim.save(filename, writer=writer, dpi=200)

        return anim

    def run(self, evaluate):
        """
        Run 1 episode
        """
        print("########## " + self.agent_name + " is running ##########")
        self.list_rewards = pd.DataFrame (columns = [self.agent_name])
        self.list_training_best_performance = pd.DataFrame (columns = [self.agent_name])
        self.list_test_best_performance = pd.DataFrame (columns = [self.agent_name])

        self.list_ground_truth = []
        self.list_evaluation_values = []
        self.list_evaluation_states = []
        self.list_evaluation_captions = []
        if self.env.dataset == "artificial":
            self.list_loss_evaluation = pd.DataFrame (columns = [str(self.env.get_algo_name_by_index_artificial(i)) for i in range(self.env.nA)])
        else:
            self.list_loss_evaluation = pd.DataFrame (columns = [str(self.env.get_algo_name_by_index(i)) for i in range(self.env.nA)])


        self.list_loss_evaluation['time_step'] = ''

        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []

        self.last_action = None
        self.number_of_switching = 0

        ep_reward = 0.0         # Reward for this episode
        done = False          # Whether the game is finished
        loss = 0.0
        episode_states = []
        episode_actions = []
        episode_rewards = []

        if self.agent_name == "average_rank" and not evaluate:
            self.precompute_rank()

        if self.agent_name == "best_on_samples":
            self.init()

        if self.agent_name == "freeze_thaw":
            self.initialize()
            self.remaining_time = self.env.max_number_of_steps

        # Get the current state
        state = self.env.observe().clone().to(self.device, dtype=torch.float)
        while not done:
            episode_states.append(state)
            with torch.no_grad():
                # Get and execute the next action for the current state
                action = self.select_action(state, evaluate)
                if self.last_action!=None and self.last_action!=action:
                    self.number_of_switching += 1
                next_state, reward, done, info = self.env.step(action)
                ep_reward = ep_reward + reward.item()
                if self.agent_name == "freeze_thaw":
                    self.think(action, next_state)

            episode_actions.append(action)
            episode_rewards.append(reward)
            self.last_action = copy.deepcopy(action)

            # Start training once the size of the buffer greater than the batch size
            if not evaluate and self.agent_name!="battleship_baseline" and self.agent_name!="random" and self.agent_name!="battleship_baseline" and self.agent_name!="learning_curve_baseline" and self.agent_name != "average_rank" and self.agent_name != "best_on_samples":
                # Save what the agent just learnt to the experience buffer.
                self.buffer.add(state, action, reward, next_state, done)
                if self.agent_name!="ppo":
                    loss = self.train()
                else:
                    loss = 0.0

            if evaluate:
                action = action.cpu().detach().numpy()
                if self.env.freely_moving:
                    x,y = np.unravel_index(action, (self.env.number_of_rows,self.env.number_of_columns))
                    evaluation_caption = "Agent: " + self.agent_name  + " Max number of steps: " + str(self.env.max_number_of_steps) + "  Step: " + str(self.env.current_number_of_steps) + "  Next Action: " + "(" + str(x) + "," + str(y) + ")" + "  Reward: " + str(reward.item()) + "  Done: " + str(bool(done.item()))
                else:
                    if action==0: #stand still
                        action_name = 'stand still'
                    elif action==1: #up
                        action_name = 'up'
                    elif action==2: #down
                        action_name = 'down'
                    elif action==3: #left
                        action_name = 'left'
                    elif action==4: #right
                        action_name = 'right'

                    evaluation_caption = "Agent: " + self.agent_name  + "  Step: " + str(self.env.current_number_of_steps-1) + "  Next Action: " + action_name + "  Reward: " + str(reward.item()) + "  Done: " + str(bool(done.item()))

                if self.env.env_name == "battleship":
                    self.list_ground_truth.append(self.env.ships_board)
                self.list_evaluation_states.append(state[0].clone())
                self.list_evaluation_captions.append(evaluation_caption)
                if self.env.dataset=="artificial":
                    self.list_loss_evaluation = self.list_loss_evaluation.append(pd.DataFrame({'time_step': math.ceil(next_state[1, x, y].item()*self.env.max_number_of_steps-1), str(self.env.get_algo_name_by_index_artificial(int(action))):next_state[0, x, y].item()}, index=[0]), ignore_index=True)
                else:
                    self.list_loss_evaluation = self.list_loss_evaluation.append(pd.DataFrame({'time_step': math.ceil(next_state[1, x, y].item()*self.env.max_number_of_steps-1), str(self.env.get_algo_name_by_index(int(action))):next_state[0, x, y].item()}, index=[0]), ignore_index=True)

            state = next_state.clone().to(self.device, dtype=torch.float)


        if evaluate:
            # For the last step
            with torch.no_grad():
                action = self.select_action(state, evaluate)
            last_evaluation_caption = "Agent: " + self.agent_name  + "  Step: " + str(self.env.current_number_of_steps) + "  Next Action: " + "None" + "  Reward: " + "None" + "  Done: " + str(bool(done.item()))
            if self.env.env_name == "battleship":
                self.list_ground_truth.append(self.env.ships_board)
            self.list_evaluation_states.append(state[0].clone())
            self.list_evaluation_captions.append(last_evaluation_caption)

        if self.env.env_name == "learning_curve":
            if evaluate:
                self.list_test_best_performance = self.list_test_best_performance.append({self.agent_name:self.env.current_max_performance}, ignore_index=True)
            else:
                self.list_training_best_performance = self.list_training_best_performance.append({self.agent_name:self.env.current_max_performance}, ignore_index=True)
        else:
            self.list_rewards = self.list_rewards.append({self.agent_name:ep_reward}, ignore_index=True)

        # Epsilon is decayed since the agent is getting more and more knowledge
        if self.epsilon != None and self.epsilon_decay_rate != None and not evaluate:
            self.epsilon = self.epsilon * self.epsilon_decay_rate

        # For PPO
        self.many_episode_states.append(episode_states)
        self.many_episode_actions.append(episode_actions)
        self.many_episode_rewards.append(episode_rewards)
        if self.agent_name=="ppo" and not evaluate:
            loss = self.train()

        # Print log
        if self.env.env_name == "learning_curve":
            print(f'Agent: {self.agent_name} . Number of steps to finish: {self.env.current_number_of_steps} . 'f'Reward: {ep_reward}')
            print("Epsilon:", self.epsilon)
        else:
            print(f'Agent: {self.agent_name} . Number of steps to finish: {self.env.current_number_of_steps} . 'f'Reward: {ep_reward}')
            print("Epsilon:", self.epsilon)

        self.switching_frequency = pd.DataFrame (columns = [self.agent_name])
        self.accumulated_reward = pd.DataFrame (columns = [self.agent_name])

        self.switching_frequency = self.switching_frequency.append({self.agent_name:self.number_of_switching/ (self.env.max_number_of_steps/self.env.time_for_each_action)}, ignore_index=True)
        self.accumulated_reward = self.accumulated_reward.append({self.agent_name:ep_reward}, ignore_index=True)

        return self.accumulated_reward, self.switching_frequency, #self.list_test_best_performance
