import sys, getopt
import os
from gym.spaces import Discrete
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import spaces
import torch
import random
from collections import deque
import pandas as pd
from nn_builder.pytorch.CNN import CNN
from nn_builder.pytorch.RNN import RNN
import pandas as pd
import seaborn as sns
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions.beta import Beta
import random
import matplotlib.animation
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from matplotlib.animation import PillowWriter
import copy
import json
from collections import defaultdict
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import operator
from sklearn.model_selection import KFold
import pytz
import time
import scipy.stats
import pickle
from envs.MetaREVEALEnv import MetaREVEALEnv
from helpers import *
from agents.average_rank import Average_Rank
from agents.base_agent import Base_Agent
from agents.best_on_samples import Best_On_Samples
from agents.ddqn import DDQN
from agents.freeze_thaw import Freeze_Thaw
from agents.ppo import PPO_Discrete
from agents.random_search import Random_Agent
from agents.sac import SAC_Discrete
random.seed(0)
sns.set_theme(style="ticks", font_scale=1.5)
rc('animation', html='jshtml')
np.seterr(divide = 'ignore')

def train():
    """
    Run agents on training folds
    """
    global time_budgets
    episode = 0
    for _, current_dataset_name in enumerate(learning_curve_env.training_data):
        print("Episode " + str(episode) + " / " + str(len(learning_curve_env.training_data)))
        evaluate=False
        T = random.choice(time_budgets)

        learning_curve_env.current_dataset_name = current_dataset_name
        learning_curve_env.current_dataset = learning_curve_env.training_data[current_dataset_name]
        print("learning_curve_env.current_dataset_name=", learning_curve_env.current_dataset_name)

        learning_curve_env.reset(reset_goals=False, training=True, max_number_of_steps=T)
        baseline_average_rank.run(evaluate=evaluate)

        learning_curve_env.reset(reset_goals=False, training=True, max_number_of_steps=T)
        agent_ddqn.run(evaluate=evaluate)

        learning_curve_env.reset(reset_goals=False, training=True, max_number_of_steps=T)
        agent_sac.run(evaluate=evaluate)

        learning_curve_env.reset(reset_goals=False, training=True, max_number_of_steps=T)
        agent_ppo.run(evaluate=evaluate)

        # break

        episode = episode + 1

def test():
    """
    Run agents on test folds
    """
    # DFs for scores
    df_temp_random = pd.DataFrame(columns = ['random'])
    df_test_results_baseline_best_on_samples = pd.DataFrame(columns = ['best_on_samples'])
    df_test_results_baseline_average_rank = pd.DataFrame(columns = ['average_rank'])
    df_test_results_random = pd.DataFrame(columns = ['random'])
    df_test_results_ddqn = pd.DataFrame(columns = ['ddqn'])
    df_test_results_sac = pd.DataFrame(columns = ['sac'])
    df_test_results_ppo = pd.DataFrame(columns = ['ppo'])
    df_test_results_freeze_thaw = pd.DataFrame(columns = ['freeze_thaw'])

    # DFs for switching frequency
    df_SF_temp_random = pd.DataFrame(columns = ['random'])
    df_SF_test_results_baseline_best_on_samples = pd.DataFrame(columns = ['best_on_samples'])
    df_SF_test_results_baseline_average_rank = pd.DataFrame(columns = ['average_rank'])
    df_SF_test_results_random = pd.DataFrame(columns = ['random'])
    df_SF_test_results_ddqn = pd.DataFrame(columns = ['ddqn'])
    df_SF_test_results_sac = pd.DataFrame(columns = ['sac'])
    df_SF_test_results_ppo = pd.DataFrame(columns = ['ppo'])
    df_SF_test_results_freeze_thaw = pd.DataFrame(columns = ['freeze_thaw'])

    global time_budgets
    episode = 0
    for _, current_dataset_name in enumerate(learning_curve_env.test_data):
        print("Episode " + str(episode) + " / " + str(len(learning_curve_env.test_data)))
        evaluate = True
        T = random.choice(time_budgets)

        learning_curve_env.current_dataset_name = current_dataset_name
        learning_curve_env.current_dataset = learning_curve_env.test_data[current_dataset_name]
        print("learning_curve_env.current_dataset_name=", learning_curve_env.current_dataset_name)

        learning_curve_env.reset(reset_goals=False, training=False, max_number_of_steps=T)
        score, switching_frequency = baseline_best_on_samples.run(evaluate=evaluate)
        df_test_results_baseline_best_on_samples = df_test_results_baseline_best_on_samples.append(score)
        df_SF_test_results_baseline_best_on_samples = df_SF_test_results_baseline_best_on_samples.append(switching_frequency)

        learning_curve_env.reset(reset_goals=False, training=False, max_number_of_steps=T)
        score, switching_frequency = baseline_average_rank.run(evaluate=evaluate)
        df_test_results_baseline_average_rank = df_test_results_baseline_average_rank.append(score)
        df_SF_test_results_baseline_average_rank = df_SF_test_results_baseline_average_rank.append(switching_frequency)

        for i in range(5):
            learning_curve_env.reset(reset_goals=False, training=False, max_number_of_steps=T)
            score, switching_frequency = agent_random.run(evaluate=evaluate)
            df_temp_random = df_temp_random.append(score)
            df_SF_temp_random = df_SF_temp_random.append(switching_frequency)

        df_test_results_random = df_test_results_random.append(df_temp_random.mean(axis=0).to_frame().T)
        df_SF_test_results_random = df_SF_test_results_random.append(df_SF_temp_random.mean(axis=0).to_frame().T)

        learning_curve_env.reset(reset_goals=False, training=False, max_number_of_steps=T)
        score, switching_frequency = agent_freeze_thaw.run(evaluate=evaluate)
        df_test_results_freeze_thaw = df_test_results_freeze_thaw.append(score)
        df_SF_test_results_freeze_thaw = df_SF_test_results_freeze_thaw.append(switching_frequency)

        learning_curve_env.reset(reset_goals=True, training=False, max_number_of_steps=T)
        score, switching_frequency = agent_ddqn.run(evaluate=evaluate)
        df_test_results_ddqn = df_test_results_ddqn.append(score)
        df_SF_test_results_ddqn = df_SF_test_results_ddqn.append(switching_frequency)

        learning_curve_env.reset(reset_goals=False, training=False, max_number_of_steps=T)
        score, switching_frequency = agent_sac.run(evaluate=evaluate)
        df_test_results_sac = df_test_results_sac.append(score)
        df_SF_test_results_sac = df_SF_test_results_sac.append(switching_frequency)

        learning_curve_env.reset(reset_goals=False, training=False, max_number_of_steps=T)
        score, switching_frequency = agent_ppo.run(evaluate=evaluate)
        df_test_results_ppo = df_test_results_ppo.append(score)
        df_SF_test_results_ppo = df_SF_test_results_ppo.append(switching_frequency)

        episode = episode + 1

        # break

#     DFs for scores
    df_test_results_baseline_best_on_samples = pd.DataFrame(df_test_results_baseline_best_on_samples.mean(axis=0))
    df_test_results_baseline_best_on_samples.reset_index(drop=True, inplace=True)
    df_test_results_baseline_best_on_samples.columns = ["best_on_samples"]

    df_test_results_baseline_average_rank = pd.DataFrame(df_test_results_baseline_average_rank.mean(axis=0))
    df_test_results_baseline_average_rank.reset_index(drop=True, inplace=True)
    df_test_results_baseline_average_rank.columns = ["average_rank"]

    df_test_results_random = pd.DataFrame(df_test_results_random.mean(axis=0))
    df_test_results_random.reset_index(drop=True, inplace=True)
    df_test_results_random.columns = ["random"]

    df_test_results_ddqn = pd.DataFrame(df_test_results_ddqn.mean(axis=0))
    df_test_results_ddqn.reset_index(drop=True, inplace=True)
    df_test_results_ddqn.columns = ["ddqn"]

    df_test_results_sac = pd.DataFrame(df_test_results_sac.mean(axis=0))
    df_test_results_sac.reset_index(drop=True, inplace=True)
    df_test_results_sac.columns = ["sac"]

    df_test_results_ppo = pd.DataFrame(df_test_results_ppo.mean(axis=0))
    df_test_results_ppo.reset_index(drop=True, inplace=True)
    df_test_results_ppo.columns = ["ppo"]

    df_test_results_freeze_thaw = pd.DataFrame(df_test_results_freeze_thaw.mean(axis=0))
    df_test_results_freeze_thaw.reset_index(drop=True, inplace=True)
    df_test_results_freeze_thaw.columns = ["freeze_thaw"]

    # DFs for switching frequency
    df_SF_test_results_baseline_best_on_samples = pd.DataFrame(df_SF_test_results_baseline_best_on_samples.mean(axis=0))
    df_SF_test_results_baseline_best_on_samples.reset_index(drop=True, inplace=True)
    df_SF_test_results_baseline_best_on_samples.columns = ["best_on_samples"]

    df_SF_test_results_baseline_average_rank = pd.DataFrame(df_SF_test_results_baseline_average_rank.mean(axis=0))
    df_SF_test_results_baseline_average_rank.reset_index(drop=True, inplace=True)
    df_SF_test_results_baseline_average_rank.columns = ["average_rank"]

    df_SF_test_results_random = pd.DataFrame(df_SF_test_results_random.mean(axis=0))
    df_SF_test_results_random.reset_index(drop=True, inplace=True)
    df_SF_test_results_random.columns = ["random"]

    df_SF_test_results_ddqn = pd.DataFrame(df_SF_test_results_ddqn.mean(axis=0))
    df_SF_test_results_ddqn.reset_index(drop=True, inplace=True)
    df_SF_test_results_ddqn.columns = ["ddqn"]

    df_SF_test_results_sac = pd.DataFrame(df_SF_test_results_sac.mean(axis=0))
    df_SF_test_results_sac.reset_index(drop=True, inplace=True)
    df_SF_test_results_sac.columns = ["sac"]

    df_SF_test_results_ppo = pd.DataFrame(df_SF_test_results_ppo.mean(axis=0))
    df_SF_test_results_ppo.reset_index(drop=True, inplace=True)
    df_SF_test_results_ppo.columns = ["ppo"]

    df_SF_test_results_freeze_thaw = pd.DataFrame(df_SF_test_results_freeze_thaw.mean(axis=0))
    df_SF_test_results_freeze_thaw.reset_index(drop=True, inplace=True)
    df_SF_test_results_freeze_thaw.columns = ["freeze_thaw"]

    return df_test_results_baseline_best_on_samples, df_test_results_baseline_average_rank, df_test_results_random, \
            df_test_results_ddqn,df_test_results_sac, df_test_results_ppo, df_test_results_freeze_thaw,\
            df_SF_test_results_baseline_best_on_samples, df_SF_test_results_baseline_average_rank, df_SF_test_results_random, \
            df_SF_test_results_ddqn, df_SF_test_results_sac, df_SF_test_results_ppo, df_SF_test_results_freeze_thaw

def load_models(baseline_average_rank, agent_ddqn, agent_sac, agent_ppo, models_dir):
    """
    Load saved models for each agent
    """

    #Average rank
    with open(models_dir+'/baseline_average_rank.pkl', 'rb') as f:
        baseline_average_rank.all_episodes_rank = pickle.load(f)

    # DDQN
    with open(models_dir+'/agent_ddqn_main_dqn.pth', 'rb') as f:
        agent_ddqn.main_dqn.load_state_dict(torch.load(f))
    with open(models_dir+'/agent_ddqn_target_dqn.pth', 'rb') as f:
        agent_ddqn.target_dqn.load_state_dict(torch.load(f))

    # PPO
    with open(models_dir+'/agent_ppo_policy_old.pth', 'rb') as f:
        agent_ppo.policy_old.load_state_dict(torch.load(f))
    with open(models_dir+'/agent_ppo_policy_new.pth', 'rb') as f:
        agent_ppo.policy_new.load_state_dict(torch.load(f))

    # SAC
    with open(models_dir+'/agent_sac_actor_local.pth', 'rb') as f:
        agent_sac.actor_local.load_state_dict(torch.load(f))
    with open(models_dir+'/agent_sac_critic_local_2.pth', 'rb') as f:
        agent_sac.critic_local_2.load_state_dict(torch.load(f))
    with open(models_dir+'/agent_sac_critic_local.pth', 'rb') as f:
        agent_sac.critic_local.load_state_dict(torch.load(f))
    with open(models_dir+'/agent_sac_critic_target_2.pth', 'rb') as f:
        agent_sac.critic_target_2.load_state_dict(torch.load(f))
    with open(models_dir+'/agent_sac_critic_target.pth', 'rb') as f:
        agent_sac.critic_target.load_state_dict(torch.load(f))

    return baseline_average_rank, agent_ddqn, agent_sac, agent_ppo

###########################################
#############  MAIN FUNCTION  #############
###########################################
if __name__ == "__main__":
    # Parse command line arguments
    dataset = ''
    type_of_learning = ''
    load_trained_models = None
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hd:l:m:",["meta_dataset=","learning_style=", "load_trained_mode="])
    except getopt.GetoptError:
        print('main.py -d <meta-dataset> -l <learning-type> -m <load-trained-models>')
        sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
           print('main.py -d <meta-dataset> -l <learning-type> -m <load-trained-models>')
           sys.exit()
       elif opt in ("-d","meta_dataset"):
           dataset = arg
       elif opt in ("-l", "learning_style"):
           type_of_learning = arg
       elif opt in ("-m", "load_trained_mode"):
           load_trained_models = arg

    # Prepare output dir and saved models dir
    time = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    delta_t = 10
    number_of_folds = 4
    output_dir = "outputs/"
    output_dir = output_dir + time + "/"
    os.mkdir(output_dir)
    trained_models_dir = "trained_models/" + dataset + "/" + type_of_learning + "/"

    list_models_for_all_folds = []
    for folder in os.listdir(trained_models_dir):
        if folder.startswith("saved_models"):
            list_models_for_all_folds.append(folder)

    # Setup meta-datasets
    if dataset == "artificial":
        number_of_datasets = 100
        number_of_columns = 20
        epsilon_decay_rate = 0.95
        time_budgets = [500, 700, 900, 1100]

    elif dataset == "autoDL":
        number_of_datasets = 66
        number_of_columns = 13
        epsilon_decay_rate = 0.9
        time_budgets = [200, 300, 400, 500]

    # Create env
    learning_curve_env =  MetaREVEALEnv(time_awareness=False, number_of_rows=1, number_of_columns=number_of_columns, time_for_each_action=delta_t, type_of_learning = type_of_learning, dataset=dataset)

    ### Plot data clustered heatmap cluster
    # clustermap = plot_data_heatmap_cluster(learning_curve_env.list_all_learning_curves, dataset + "_metadataset")

    # Split data
    list_keys = list(learning_curve_env.list_all_learning_curves.keys())
    kf = KFold(n_splits=number_of_folds, shuffle=False)

    pd_results = pd.DataFrame()
    pd_frequency = pd.DataFrame()
    iteration = 0
    for train_index, test_index in kf.split(list_keys):
        print("Iteration " + str(iteration) + " / " + str(number_of_folds))
        pd_temp_scores = pd.DataFrame()
        pd_temp_SF = pd.DataFrame()

        # Init agents
        baseline_best_on_samples = Best_On_Samples(agent_name="best_on_samples", env=learning_curve_env, samples_time_for_each_algo=delta_t)
        baseline_average_rank = Average_Rank(agent_name="average_rank", env=learning_curve_env)
        agent_random = Random_Agent(agent_name="random", env=learning_curve_env)
        agent_ddqn = DDQN(agent_name="ddqn", env=learning_curve_env, epsilon=1, epsilon_decay_rate=epsilon_decay_rate, gamma=0.99, batch_size=256)
        agent_sac = SAC_Discrete(agent_name="sac", env=learning_curve_env, gamma=0.99, batch_size=256, automatic_entropy_tuning=False)
        agent_ppo = PPO_Discrete(agent_name="ppo", env=learning_curve_env, gamma=0.99, batch_size=256, epsilon=1, epsilon_decay_rate=epsilon_decay_rate)
        agent_freeze_thaw = Freeze_Thaw(agent_name="freeze_thaw", env=learning_curve_env)

        # Prepare train/test set
        training_indices = [list_keys[i] for i in train_index]
        learning_curve_env.training_data = dict((k, learning_curve_env.list_all_learning_curves[k]) for k in training_indices)
        testing_indices = [list_keys[i] for i in test_index]
        learning_curve_env.test_data = dict((k, learning_curve_env.list_all_learning_curves[k]) for k in testing_indices)

        # Load saved models / Train new ones
        print("load_trained_models", load_trained_models)
        if load_trained_models=="True":
            baseline_average_rank, agent_ddqn, agent_sac, agent_ppo = load_models(baseline_average_rank, agent_ddqn, agent_sac, agent_ppo, trained_models_dir+list_models_for_all_folds[iteration])
        else:
            train()
            save_models(agent_ddqn, agent_sac, agent_ppo, baseline_average_rank, output_dir)

        # Run on test folds
        df_test_results_baseline_best_on_samples, df_test_results_baseline_average_rank, df_test_results_random, \
                df_test_results_ddqn,df_test_results_sac, df_test_results_ppo, df_test_results_freeze_thaw,\
                df_SF_test_results_baseline_best_on_samples, df_SF_test_results_baseline_average_rank, df_SF_test_results_random, \
                df_SF_test_results_ddqn, df_SF_test_results_sac, df_SF_test_results_ppo, df_SF_test_results_freeze_thaw = test()


        # Prepare results
        df_test_results_baseline_best_on_samples.reset_index(drop=True, inplace=True)
        df_test_results_baseline_average_rank.reset_index(drop=True, inplace=True)
        df_test_results_random.reset_index(drop=True, inplace=True)
        df_test_results_ddqn.reset_index(drop=True, inplace=True)
        df_test_results_sac.reset_index(drop=True, inplace=True)
        df_test_results_ppo.reset_index(drop=True, inplace=True)
        df_test_results_freeze_thaw.reset_index(drop=True, inplace=True)

        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_baseline_best_on_samples], axis=1)
        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_baseline_average_rank], axis=1)
        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_random], axis=1)
        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_ddqn], axis=1)
        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_sac], axis=1)
        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_ppo], axis=1)
        pd_temp_scores = pd.concat([pd_temp_scores, df_test_results_freeze_thaw], axis=1)

        pd_results = pd.concat([pd_results, pd_temp_scores], axis=0)

        df_SF_test_results_baseline_best_on_samples.reset_index(drop=True, inplace=True)
        df_SF_test_results_baseline_average_rank.reset_index(drop=True, inplace=True)
        df_SF_test_results_random.reset_index(drop=True, inplace=True)
        df_SF_test_results_ddqn.reset_index(drop=True, inplace=True)
        df_SF_test_results_sac.reset_index(drop=True, inplace=True)
        df_SF_test_results_ppo.reset_index(drop=True, inplace=True)
        df_SF_test_results_freeze_thaw.reset_index(drop=True, inplace=True)

        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_baseline_best_on_samples], axis=1)
        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_baseline_average_rank], axis=1)
        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_random], axis=1)
        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_ddqn], axis=1)
        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_sac], axis=1)
        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_ppo], axis=1)
        pd_temp_SF = pd.concat([pd_temp_SF, df_SF_test_results_freeze_thaw], axis=1)

        pd_frequency = pd.concat([pd_frequency, pd_temp_SF], axis=0)

        iteration += 1
        # break

    pd_results = pd_results.reindex(pd_results.mean().sort_values(ascending=False).index, axis=1)
    pd_frequency = pd_frequency.reindex(pd_frequency.mean().sort_values(ascending=False).index, axis=1)

    # Average accumulated reward
    print("[+] Average Accumulated Reward")
    print(pd_results.mean())
    print("[+] Standard Deviation")
    print(pd_results.std())
    date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    filename= type_of_learning + "_" + dataset + "_t0=" + str(delta_t) + "scores_" + date + ".csv"
    pd_results.to_csv(output_dir+filename, header=True, index=False)

    # Average switching frequency
    print("[+] Average Switching Frequency")
    print(pd_frequency.mean())
    print("[+] Standard Deviation")
    print(pd_frequency.std())
    date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    filename= type_of_learning + "_" + dataset + "_t0=" + str(delta_t) + "frequency_" + date + ".csv"
    pd_frequency.to_csv(output_dir+filename, header=True, index=False)

    # Plot Average Accumulated Reward
    plot_results_scatterplot(pd_results, pd_frequency, download=True, filename= output_dir+type_of_learning + "_" + dataset)

    # Plot correlations between Average Accumulated Reward and Average Switching Frequency
    plot_results_barplot(pd_results, rolling = 1, max_value=1.1, download=True, filename = output_dir+type_of_learning + "_" + dataset)
