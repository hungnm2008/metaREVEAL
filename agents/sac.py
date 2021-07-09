

class SAC_Discrete(Base_Agent):
    """ Soft Actor Critic agent for discrete action space
    """
    def __init__(self, agent_name, env, gamma, batch_size, automatic_entropy_tuning):
        super().__init__(agent_name, env)

        self.gamma = gamma # The weight for future rewards
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.buffer = Replay()
        self.L1loss = torch.nn.SmoothL1Loss()
        self.mse = torch.nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.env.nA)) * 0.98
#             self.target_entropy = 0.2
#             self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
#             self.alpha = self.log_alpha.exp()
            self.alpha = torch.ones(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.alpha], lr=1e-4, eps=1e-4)
        else:
            self.alpha = 0.5
            self.add_extra_noise = False
            self.do_evaluation_iterations = False


        self.critic_local = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
#                             output_activation="softmax",
#                             dropout=0.5,
                            initialiser="xavier",
                            batch_norm=False)
        self.critic_target = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
#                             output_activation="softmax", dropout=0.0,
                            initialiser="xavier",
                            batch_norm=False)
        self.critic_local_2 = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
#                             output_activation="softmax", dropout=0.0,
                            initialiser="xavier",
                            batch_norm=False)
        self.critic_target_2 = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
#                             output_activation="softmax", dropout=0.0,
                            initialiser="xavier",
                            batch_norm=False)
        self.actor_local = CNN(input_dim=(self.env.number_of_state_channels, self.env.number_of_rows, self.env.number_of_columns), layers_info=[
                            ["conv", self.env.number_of_state_channels, 1, 1, 1],
                            ["linear", 64],
                            ["linear", 32],
                            ["linear", self.env.nA]],
                            hidden_activations="relu",
                            output_activation="softmax",
#                             dropout=0.0,
                            initialiser="xavier",
                            batch_norm=False)

        self.critic_local.to(self.device)
        self.critic_target.to(self.device)
        self.critic_local_2.to(self.device)
        self.critic_target_2.to(self.device)
        self.actor_local.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr=1e-4)

        copy_model_over(self.critic_local, self.critic_target)
        copy_model_over(self.critic_local_2, self.critic_target_2)

    def create_actor_distribution(self, action_probabilities):
        """Creates a distribution that the actor can then use to randomly draw actions"""

        action_distribution = Categorical(action_probabilities)  # this creates a distribution to sample from
        return action_distribution


    def produce_action_and_action_info(self, state, evaluate):
        '''
        Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action
        '''
        action_probabilities = self.actor_local(state)

        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = self.create_actor_distribution(action_probabilities)

        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)

        return action, (action_probabilities,log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        '''
        Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
        term is taken into account
        '''
        with torch.no_grad():
#             if self.env.env_name == "battleship":
#                 ravel = torch.tensor([[self.env.number_of_columns*1.0], [1.0]], dtype=torch.float64).to(self.device, dtype=torch.float)
#             elif self.env.env_name == "segment":
#                 ravel = torch.tensor([[self.env.number_of_rows * self.env.number_of_columns * 1.0], [self.env.number_of_columns*1.0], [1.0]], dtype=torch.float64).to(self.device, dtype=torch.float)

#             action_batch = torch.matmul(action_batch,ravel)

            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch, evaluate=False)

            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)

            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)

#             print("torch.min(qf1_next_target, qf2_next_target)=", torch.min(qf1_next_target, qf2_next_target))
#             print("-self.alpha * log_action_probabilities", -self.alpha * log_action_probabilities)

            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + torch.matmul(done_batch, self.gamma * min_qf_next_target)


        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())

        qf1_loss = self.mse(qf1, next_q_value)
        qf2_loss = self.mse(qf2, next_q_value)

        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        '''
        Calculates the loss for the actor. This loss includes the additional entropy term
        '''
        _, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch, evaluate=False)

        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi) #size [256, 9]

        inside_term = self.alpha * log_action_probabilities - min_qf_pi   #size [256, 9]


        entropies = torch.sum(log_action_probabilities * action_probabilities, dim=1)

        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()

        return policy_loss, entropies



    def select_action(self, state, evaluate):
        state = [state]
        state = torch.stack(state)
        state = state.to(self.device, dtype=torch.float)
        if not evaluate:
            action, (action_probabilities,log_action_probabilities), _ = self.produce_action_and_action_info(state, evaluate)
        else:
            _, (action_probabilities,log_action_probabilities), _ = self.produce_action_and_action_info(state, evaluate)
            action = torch.argmax(action_probabilities).item()
#             print("action_probabilities=", action_probabilities)

        if evaluate:
            if self.env.freely_moving:
                self.list_evaluation_values.append(action_probabilities.reshape(self.env.number_of_rows, self.env.number_of_columns))
            else:
                self.list_evaluation_values.append(action_probabilities.reshape(1, self.env.nA))

        return torch.tensor(action)
#         if self.env.env_name == "battleship":
#             return torch.tensor(np.unravel_index(action.item(), (self.env.number_of_rows, self.env.number_of_columns)))
#         elif self.env.env_name == "segment":
#             return torch.tensor(np.unravel_index(action.item(), (self.env.number_of_action_channels, self.env.number_of_rows, self.env.number_of_columns)))


    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=5, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            loss.backward(retain_graph=retain_graph)
#             for net in network:
#                 for param in net.parameters():
#     #                 # Clip the target to avoid exploding gradients
#                     param.grad.data.clamp_(-1e-6,1e-6)

            if clipping_norm is not None:
                for net in network:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
#             optimizer.step()

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def train(self):
        """
        Train the network with a batch of samples
        :param states: The state before taking the action
        :param actions: action taken
        :param rewards: Reward for taking that action
        :param next_states: The state that the agent enters after taking the action
        :return loss: the loss value after training the batch of samples
        """
        if len(self.buffer) >= self.batch_size:
            with torch.no_grad():
                states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                states = torch.stack(states).to(self.device, dtype=torch.float)
                actions = torch.stack(actions).to(self.device, dtype=torch.float)
#                 print("rewards=", rewards)
                rewards = torch.stack(rewards).to(self.device, dtype=torch.float)
                next_states = torch.stack(next_states).to(self.device, dtype=torch.float)
                rewards = torch.reshape(rewards, (self.batch_size, 1))

                dones = torch.stack(dones).to(self.device, dtype=torch.long)
                nonzero_indices = torch.nonzero(dones).reshape(-1).tolist()
                dones_mask = torch.eye(self.batch_size)
                for index in nonzero_indices:
                    dones_mask[index,index] = 0
                dones_mask = dones_mask.to(self.device, dtype=torch.float)  # size [256, 256]

            #Updates the parameters for both critics
            qf1_loss, qf2_loss = self.calculate_critic_losses(states, actions, rewards, next_states, dones_mask)
            self.take_optimisation_step(self.critic_optimizer, self.critic_local, qf1_loss,None)
            self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, qf2_loss,None)
            self.critic_optimizer.step()
            self.critic_optimizer_2.step()
            self.soft_update_of_target_network(self.critic_local, self.critic_target,tau=1e-3)
            self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,tau=1e-3)

            # update_actor_parameters
            policy_loss, log_pi = self.calculate_actor_loss(states)
            self.take_optimisation_step(self.actor_optimizer, self.actor_local, policy_loss,None)
            self.actor_optimizer.step()

            if self.automatic_entropy_tuning:
                alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
            else:
                alpha_loss = None


            if alpha_loss is not None:
                self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
                self.alpha_optim.step()
#                 self.alpha = self.log_alpha.exp()
            return qf1_loss
        return 0
