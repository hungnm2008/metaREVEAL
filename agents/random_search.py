class Random_Agent(Base_Agent):
    """
    Model a random agent
    """
    def __init__(self, agent_name, env):
        super().__init__(agent_name, env)

    def select_action(self, state, evaluate=False):
        """
          Return an random action
          :param state: the current state
          :return action: next action to take
        """
        random.seed(time.time())
        if self.env.env_name == "segment":
            random_action = torch.tensor([random.randint(0,self.env.number_of_action_channels-1), random.randint(0,self.env.number_of_rows-1), random.randint(0,self.env.number_of_columns-1)])
        elif self.env.env_name == "battleship":
            random_action = torch.tensor([random.randint(0,self.env.number_of_rows-1), random.randint(0,self.env.number_of_columns-1)])
        else:
            random_action = torch.tensor(random.randint(0,self.env.nA-1))

        return random_action
