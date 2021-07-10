##### HELPER FUNCTIONS #####
import time
import datetime
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
import os
import torch
import pickle
from matplotlib.pyplot import figure
import numpy as np

def copy_model_over(from_model, to_model):
        """
        Copies model parameters from from_model to to_model
        """
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

def plot_results_barplot(pd_results, rolling, max_value, download=False, filename=""):
    """
    Bar plot of Average accumulated reward
    """
    figure(figsize=(6, 6)) #dpi=150
    palette ={'freeze_thaw': "darkorange", 'ppo': "steelblue", 'sac': "steelblue", 'ddqn': "steelblue", "average_rank": "darkorange", 'random':'darkorange', 'best_on_samples':'darkorange'}
    ax = sns.barplot(data=pd_results, ci='sd', palette=palette)
    plt.title(filename) #fontsize=4
    plt.xlabel('Agent') #fontsize=4
    plt.ylabel('Average accumulated reward') #fontsize=4
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(0,0,1,1,0,0)
    plt.xticks(rotation=75)
    plt.ylim([0, max_value])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    date = datetime.datetime.now(pytz.timezone('Europe/Paris')).strftime("%Y_%m_%d-%I_%M_%S_%p")
    plt.savefig(filename+ "_barplot_" + date + ".svg", dpi=1200, bbox_inches='tight', pad_inches = 0, format='svg')
    filename = filename + "_scores_" + date + ".csv"
    if download:
        pd_results.to_csv(filename, header=True, index=False)

def plot_results_scatterplot(pd_results, pd_frequency, download=False, filename=""):
    """
    Scatter plot of correlation between Average accumulated reward and Average switching frequency
    """
    figure(figsize=(6, 6))
    plt.title(filename)
    palette ={'RL': "steelblue","Baseline": "darkorange"}
    a = pd.DataFrame(pd_results.mean())
    a.columns = ["Average accumulated reward"]
    b = pd.DataFrame(pd_frequency.mean())
    b.columns = ["Average switching frequency"]
    c = pd.merge(a, b, left_index=True, right_index=True)
    c.reset_index(inplace=True)
    c = c.rename(columns={"index": "Agent"})
    c['Group of Agents'] = np.where((c.Agent == 'sac') | (c.Agent == 'ddqn') | (c.Agent == 'ppo'), 'RL', 'Baseline')
    c = c.sort_values(by=['Agent'])
    print(c)

    marker_list = ['d', 'X', '>', 'p', 's', 'o', '^']

    sns.scatterplot(data=c, x="Average switching frequency", y="Average accumulated reward",  \
                    hue="Group of Agents", palette=palette,  style="Agent", markers=marker_list, \
                    s=300, edgecolor='black', alpha=0.5)
    plt.ylim([0, 1])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.xlim([-0.05, 1])
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(0,0,1,1,0,0)

    date = datetime.datetime.now(pytz.timezone('Europe/Paris')).strftime("%Y_%m_%d-%I_%M_%S_%p")
    plt.savefig(filename+ "_scatterplot_" + date + ".svg", dpi=1200, bbox_inches='tight', pad_inches = 0, format='svg')
    filename = filename + "_sscatterplot_" + date + ".csv"
    if download:
        c.to_csv(filename, header=True, index=False)

def plot_data_heatmap_cluster(list_all_learning_curves, filename=""):
    """
    Clustered heatmap of each meta-dataset
    """
    a_dict = list_all_learning_curves
    df = pd.DataFrame(columns = [str(i) for i in range(20)])
    for key, row in a_dict.items():
        new_row = {str(i):row[str(i)].a for i in range(20)}
        df = df.append(new_row, ignore_index=True)
    g = sns.clustermap(df, row_cluster=True, col_cluster=True, yticklabels=0, xticklabels=0, cbar_kws={'shrink': 0.6}, cmap="Blues")
    ax = g.ax_heatmap
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Algorithm")
    g.fig.set_size_inches(8,8)
    g.fig.suptitle(filename, y=1)
    plt.show()
    date = datetime.datetime.now(pytz.timezone('Europe/Paris')).strftime("%Y_%m_%d-%I_%M_%S_%p")
    g.savefig(filename+ "_heatmap_" + date + ".svg", dpi=1200, bbox_inches='tight', pad_inches = 0, format='svg')
    return g

def save_models(agent_ddqn, agent_sac, agent_ppo, baseline_average_rank, output_dir):
    """
    Save trained models
    """
    date = datetime.datetime.now(pytz.timezone('Europe/Paris')).strftime("%Y_%m_%d-%I_%M_%S_%p")
    path = output_dir + "saved_models_" + date
    os.mkdir(path)
    #DDQN
    torch.save(agent_ddqn.main_dqn.state_dict(), path + "/agent_ddqn_main_dqn.pth")
    torch.save(agent_ddqn.target_dqn.state_dict(), path + "/agent_ddqn_target_dqn.pth")

    #SAC
    torch.save(agent_sac.critic_local.state_dict(), path + "/agent_sac_critic_local.pth")
    torch.save(agent_sac.critic_target.state_dict(), path + "/agent_sac_critic_target.pth")
    torch.save(agent_sac.critic_local_2.state_dict(), path + "/agent_sac_critic_local_2.pth")
    torch.save(agent_sac.critic_target_2.state_dict(), path + "/agent_sac_critic_target_2.pth")
    torch.save(agent_sac.actor_local.state_dict(), path + "/agent_sac_actor_local.pth")

    #PPO
    torch.save(agent_ppo.policy_new.state_dict(), path + "/agent_ppo_policy_new.pth")
    torch.save(agent_ppo.policy_old.state_dict(), path + "/agent_ppo_policy_old.pth")

    with open(path + '/baseline_average_rank.pkl', 'wb') as f:
        pickle.dump(baseline_average_rank.all_episodes_rank, f)
