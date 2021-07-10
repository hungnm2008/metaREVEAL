from agents.base_agent import *
from nn_builder.pytorch.CNN import CNN
from helpers import *
import random
import numpy as np

class DDQN(Base_Agent):
    """
    Model Double DQN agent
    """
    def __init__(self, agent_name, env, epsilon, epsilon_decay_rate, gamma, batch_size):
        super().__init__(agent_name, env)
        self.epsilon = epsilon # For deciding exploitation or exploration
        self.epsilon_decay_rate = epsilon_decay_rate # Epsilon is decayed after each episode with a fixed rate
        self.gamma = gamma # The weight for future rewards
        self.batch_size = batch_size

        self.buffer = Replay()             # Experience Buffer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.main_dqn = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns),
                            layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
                            initialiser="xavier",
                            batch_norm=False)

        self.target_dqn = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns),
                            layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
                            initialiser="xavier",
                            batch_norm=False)

        # Send models to GPU
        self.main_dqn.to(self.device)
        self.target_dqn.to(self.device)

         # Optimizer and Loss function
        self.optimizer = torch.optim.Adam(self.main_dqn.parameters(), lr=1e-4)
        self.mse = torch.nn.MSELoss()
        self.L1 = torch.nn.SmoothL1Loss()

    def select_action(self, state, evaluate):
        """
          Return an action to take based on epsilon (greedy or random action)
          :param state: the current state
          :return action: next action to take
        """
        random_number = np.random.uniform()
        if random_number < self.epsilon and evaluate==False:
            # Random action
            return torch.tensor(random.randint(0,self.env.nA-1))

        else:
            # Greedy action
            state = [state]
            state = torch.stack(state)
            state = state.to(self.device, dtype=torch.float)
            q_values = self.main_dqn(state)
            argmax = torch.argmax(q_values).item()

            if evaluate:
                if self.env.freely_moving:
                    self.list_evaluation_values.append(q_values.reshape(self.env.number_of_rows, self.env.number_of_columns))
                else:
                    self.list_evaluation_values.append(q_values.reshape(1, self.env.nA))

            return torch.tensor(argmax)

    def train(self):
        """
        Train the network with a batch of samples
        """
        if len(self.buffer) >= self.batch_size:
            with torch.no_grad():
                states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            # Send data to GPU
            states = torch.stack(states).to(self.device, dtype=torch.float)
            actions = torch.stack(actions).to(self.device, dtype=torch.float)
            rewards = torch.stack(rewards).to(self.device, dtype=torch.float)
            rewards = torch.reshape(rewards, (self.batch_size, 1))

            next_states = torch.stack(next_states).to(self.device, dtype=torch.float)
            dones = torch.stack(dones).to(self.device, dtype=torch.float)

            #TODO

            # Calculate target Q values using the Target Network
            selection = torch.argmax(self.main_dqn(next_states), dim = 1).unsqueeze(1)

            evaluation = self.target_dqn(next_states)
            evaluation = evaluation.gather(1, selection.long()) #size [256,1]

            #Create Done mask
            nonzero_indices = torch.nonzero(dones).reshape(-1).tolist()
            dones_mask = torch.eye(self.batch_size)
            for index in nonzero_indices:
                dones_mask[index,index] = 0
            dones_mask = dones_mask.to(self.device, dtype=torch.float)

            # Calculte target
            target = rewards + torch.matmul(dones_mask, evaluation*self.gamma)
            target = target.detach()

            # Calculate Q values using the Main Network
            if self.env.freely_moving:
                n_classes = self.env.number_of_action_channels * self.env.number_of_rows * self.env.number_of_columns
            else:
                n_classes = self.env.number_of_action_channels * 1 * self.env.nA

            n_samples = self.batch_size
            labels = torch.flatten(actions.type(torch.LongTensor), start_dim=0)
            labels_tensor = torch.as_tensor(labels)
            action_masks = torch.nn.functional.one_hot(labels_tensor, num_classes=n_classes).to(self.device, dtype=torch.float)

            q_value = action_masks * self.main_dqn(states)
            q_value = torch.sum(q_value, dim=-1).reshape((self.batch_size, 1))

            # Calculate loss
            loss = self.mse(target, q_value)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_dqn.parameters(), 5)
            self.optimizer.step()

            # Soft Copy the Main Network's weights to the Target Network
            self.soft_update_of_target_network(self.main_dqn, self.target_dqn,tau=1e-3)

            return loss
        return 0
