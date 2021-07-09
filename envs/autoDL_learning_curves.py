
class SigmoidCurve():
    def __init__(self, algo_name, a, b, c):
        self.algo_name = algo_name
        self.a = a
        self.b = b
        self.c = c

    def get_value(self, time_step):
        return min(1.0, max(0.0, self.a / (1 + math.exp(-self.b * (time_step - self.c)))))
        
class AutoDLEnv(gym.Env):
    def __init__(self, time_awareness, number_of_rows, number_of_columns, time_for_each_action,type_of_learning, dataset):
        self.env_name = "learning_curve"
        self.time_awareness = time_awareness
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.learning_curves = []
        self.time_for_each_action = time_for_each_action
        self.freely_moving = True
#         self.reward_type = reward_type
        self.type_of_learning = type_of_learning
        self.dataset = dataset

        if self.time_awareness:
            self.number_of_state_channels = 3
        else:
            self.number_of_state_channels = 2

        self.number_of_action_channels = 1
        self.nA = number_of_columns # Size of action space n*n

        if self.dataset == "autoDL":
        # Read all data from files
            self.list_all_learning_curves = defaultdict(dict)
            list_algo_names = [d for d in os.listdir('/kaggle/input/learningcurves/meta-ml_matrix_autodl/') if os.path.isdir(os.path.join('/kaggle/input/learningcurves/meta-ml_matrix_autodl/', d))]
            list_dataset_names = [d for d in os.listdir('/kaggle/input/learningcurves/meta-ml_matrix_autodl/AutoMLFreiburg/') if os.path.isdir(os.path.join('/kaggle/input/learningcurves/meta-ml_matrix_autodl/AutoMLFreiburg/', d))]
            for algo in list_algo_names:
                for dataset in list_dataset_names:
                    curve = AutoDL_Learning_Curve(algo, dataset)
                    self.list_all_learning_curves[dataset][algo] = copy.deepcopy(curve)

        elif self.dataset == "artificial":
#             self.list_all_learning_curves = defaultdict(dict)
# #             list_algo_names = [d for d in os.listdir('/kaggle/input/learningcurves/meta-ml_matrix_autodl/') if os.path.isdir(os.path.join('/kaggle/input/learningcurves/meta-ml_matrix_autodl/', d))]
#             list_algo_names = [str(d) for d in range(20)]
#             list_dataset_names = [str(d) for d in range(200)]
#             for algo in list_algo_names:
#                 for dataset in list_dataset_names:
#                     curve = SigmoidCurve(algo)
#                     self.list_all_learning_curves[dataset][algo] = copy.deepcopy(curve)
            a_path = '/kaggle/input/artificial-metadataset/a.txt'
            b_path = '/kaggle/input/artificial-metadataset/b.txt'
            c_path = '/kaggle/input/artificial-metadataset/c.txt'

            with open(a_path) as f:
                a = [[float(digit) for digit in line.split()] for line in f]
            with open(b_path) as f:
                b = [[float(digit) for digit in line.split()] for line in f]
            with open(c_path) as f:
                c = [[float(digit) for digit in line.split()] for line in f]
            self.list_all_learning_curves = defaultdict(dict)

            for i in range(100):
                for j in range(self.number_of_columns):
                    curve = SigmoidCurve(str(j), a[i][j], b[i][j], c[i][j])
                    self.list_all_learning_curves[str(i)][str(j)] = copy.deepcopy(curve)

    def get_algo_name_by_index(self, index):
        if index==0:
            return 'AutoMLFreiburg'
        elif index==1:
            return 'baseline1_linear'
        elif index==2:
            return 'baseline2_3dcnn'
        elif index==3:
            return 'baseline3_all_winners'
        elif index==4:
            return 'DeepBlueAI'
        elif index==5:
            return 'DeepWisdom'
        elif index==6:
            return 'Frozenmad'
        elif index==7:
            return 'InspurAutoDL'
        elif index==8:
            return 'Kon'
        elif index==9:
            return 'PasaNJU'
        elif index==10:
            return 'pretrained_inception'
        elif index==11:
            return 'Surromind'
        elif index==12:
            return 'TeamZhaw'

    def get_algo_name_by_index_artificial(self, index):
#         [12, 11, 17, 13, 9, 0, 10, 8, 2, 18, 3, 16, 6, 1, 4, 7, 5, 19, 14, 15]
        if index==0:
            return '12'
        elif index==1:
            return '11'
        elif index==2:
            return '17'
        elif index==3:
            return '13'
        elif index==4:
            return '9'
        elif index==5:
            return '0'
        elif index==6:
            return '10'
        elif index==7:
            return '8'
        elif index==8:
            return '2'
        elif index==9:
            return '18'
        elif index==10:
            return '3'
        elif index==11:
            return '16'
        elif index==12:
            return '6'
        elif index==13:
            return '1'
        elif index==14:
            return '4'
        elif index==15:
            return '7'
        elif index==16:
            return '5'
        elif index==17:
            return '19'
        elif index==18:
            return '14'
        elif index==19:
            return '15'

    def precompute_best_algo(self):
        for algo, curve in self.current_dataset.items():
            time_spent = 0.0
            while time_spent + self.time_for_each_action < self.max_number_of_steps:
                time_spent = time_spent + self.time_for_each_action
                if curve.get_value(time_spent)>self.max_performance:
                    self.max_performance = curve.get_value(time_spent)
                    self.best_algo = algo
                    self.best_algo_time = time_spent

    def generate_board(self, training):
        if self.dataset == "autoDL":
            if training:
                random.seed(time.time())
                self.current_dataset_name, _ = random.choice(list(self.training_data.items()))
                return self.training_data[self.current_dataset_name]
            else:
                random.seed(time.time())
                self.current_dataset_name, _ = random.choice(list(self.test_data.items()))
                return self.test_data[self.current_dataset_name]


    def reset(self, reset_goals, training, max_number_of_steps):
        self.current_number_of_steps = 0
        self.current_max_performance = 0.0
        self.max_number_of_steps = max_number_of_steps
        self.max_performance = 0.0
        self.best_algo = ''
        self.current_algo_name = ''

#         if reset_goals:
#             self.current_dataset = self.generate_board(training=training)
#             print("self.current_dataset_name=", self.current_dataset_name)

        # Reset current state
        self.s = torch.ones(self.number_of_state_channels, self.number_of_rows, self.number_of_columns)
        if self.time_awareness:
            self.s[0] = (-1.0) * self.s[0] # Current performance
            self.s[1] = 0.0 * self.s[1] # Current level of learning curve, time has spent
            self.s[2] = max_number_of_steps # Max steps
        else:
            self.s[0] = (-1.0) * self.s[0] # Current performance
            self.s[1] = 0.0 * self.s[1] # Current level of learning curve

#         self.precompute_best_algo()

        return self.s

    def calculate_reward(self, x, y):
        reward = 0.0
        performance_improvement = 0.0
        t_0 = 50

        if self.s[0,x,y].item() > self.current_max_performance:
            performance_improvement = self.s[0,x,y].item()-self.current_max_performance

        if self.type_of_learning == "any_time_learning":
#             self.reward_type = "ALC"
            normalized_t_k = math.log(1+self.current_number_of_steps/t_0)/math.log(1+self.max_number_of_steps/t_0)
            reward = reward + performance_improvement*(1-normalized_t_k)

        elif self.type_of_learning == "fixed_time_learning":
#             self.reward_type = "NAUC"
            reward = reward + performance_improvement

        self.current_max_performance = max(self.current_max_performance, self.s[0,x,y].item())
        return torch.tensor(reward)

    def update_board(self, x, y):

        # Increase current level of this learning curve by 1
        self.s[1,x,y] = ((self.s[1,x,y] * self.max_number_of_steps) + self.time_for_each_action)/self.max_number_of_steps

        # Reveal learning curve
        self.s[0,x,y] = self.current_dataset[self.current_algo_name].get_value(self.s[1,x,y]*self.max_number_of_steps)

        return self.s

    def step(self, action_):
        # Increase time count
        self.current_number_of_steps = self.current_number_of_steps + self.time_for_each_action

        # Parse action
        action = torch.tensor(np.unravel_index(action_, (self.number_of_rows, self.number_of_columns)))
        x = action[0] # x coordinate of the action
        y = action[1] # y coordinate of the action
        if self.dataset == "artificial":
            self.current_algo_name = self.get_algo_name_by_index_artificial(y)
        else:
            self.current_algo_name = self.get_algo_name_by_index(y)

        # Update board and calculate reward
        next_state = self.update_board(x,y)
        reward = self.calculate_reward(x,y)

        # Check done
        done = torch.tensor(0)
#         if self.s[0,x,y] >= self.max_performance:
#             done = torch.tensor(1)
#             reward = torch.tensor(copy.deepcopy(self.current_max_performance))
        if self.current_number_of_steps >= self.max_number_of_steps:
            done = torch.tensor(1)
#             reward = torch.tensor(copy.deepcopy(self.current_max_performance))

        return next_state, reward, done, {}

    def observe(self):
        """
          Return the current state
        """
        return self.s
