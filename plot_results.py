#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 00:26:31 2021

@author: wenminggong

plot the training results
"""


import os
import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import get_monitor_files


def plot_results(tensorboard_log, total_timesteps, n_envs, n_steps, env_name):
    # plot the results
    monitor_logs_path_list = get_monitor_files(tensorboard_log)
    total_return_array = np.zeros(total_timesteps //  (n_envs * n_steps))
    robot_return_array = np.zeros(total_timesteps //  (n_envs * n_steps))
    pref_return_array = np.zeros(total_timesteps //  (n_envs * n_steps))
    for logs_path in monitor_logs_path_list:
        with open(logs_path, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pd.read_csv(file_handler, index_col=None)
            total_return_array += data_frame['total_reward'].values
            robot_return_array += data_frame['robot_reward'].values
            pref_return_array += data_frame['pref_reward'].values
    total_return_array = total_return_array / n_envs
    robot_return_array = robot_return_array / n_envs
    pref_return_array = pref_return_array / n_envs
    
    x = np.arange(0, total_timesteps, n_envs * n_steps)
    
    plt.figure(figsize=(6,4))
    # plt.plot(x, return_array, lw=1, c = 'r', marker = 'o', label = 'ppo_total_reward')
    plt.plot(x, total_return_array, lw=0.5, c = 'r', label = 'ppo_total_reward')
    plt.plot(x, robot_return_array, lw=0.5, c = 'b', label = 'ppo_robot_reward')
    plt.plot(x, pref_return_array, lw=0.5, c = 'g', label = 'ppo_pref_reward')
    
    plt.legend()
    plt.ylabel("Average return")
    plt.xlabel("Timesteps")
    plt.ylim((-400, 200))
    plt.title("ppo in " + env_name + ' robot_reward_' + 'seed:856')
    plt.grid(axis = 'y', ls = '--')
    plt.savefig(os.path.join(tensorboard_log, 'ppo_in_'+env_name + '_robot_reward'), format='eps')
    plt.show()
    
    
def plot_results_compare(tensorboard_log_list, total_timesteps, n_envs, n_steps, env_name):
    # plot the results
    plt.figure(figsize=(6,4))
    x = np.arange(0, total_timesteps, n_envs * n_steps)
    
    color = ['r', 'g', 'b', 'blcak', 'pink', 'purple']
    
    for i in range(len(tensorboard_log_list)):
        monitor_logs_path_list = get_monitor_files(tensorboard_log_list[i])
        total_return_array = np.zeros(total_timesteps //  (n_envs * n_steps))
        robot_return_array = np.zeros(total_timesteps //  (n_envs * n_steps))
        pref_return_array = np.zeros(total_timesteps //  (n_envs * n_steps))
        for logs_path in monitor_logs_path_list:
            with open(logs_path, "rt") as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#"
                header = json.loads(first_line[1:])
                data_frame = pd.read_csv(file_handler, index_col=None)
                # total_return_array += data_frame['total_reward'].values
                # robot_return_array += data_frame['robot_reward'].values
                pref_return_array += data_frame['pref_reward'].values
        # total_return_array = total_return_array / n_envs
        # robot_return_array = robot_return_array / n_envs
        pref_return_array = pref_return_array / n_envs
        if 'total_reward' in tensorboard_log_list[i]:
            plt.plot(x, pref_return_array, lw=0.5, c = color[i], label = 'ppo_total_reward')
        else:
            plt.plot(x, pref_return_array, lw=0.5, c = color[i], label = 'ppo_robot_reward')
    
    plt.legend()
    plt.ylabel("Average return")
    plt.xlabel("Timesteps")
    plt.ylim((-200, 50))
    plt.title("ppo in " + env_name + ' pref_reward')
    plt.grid(axis = 'y', ls = '--')
    plt.savefig(os.path.join(os.path.join('logs/PPO', 'FeedingSeparateRewardBaxter-v1-version9'), 'ppo_in_'+env_name + '_pref_reward_856'), format='eps')
    plt.show()
    
    
if __name__ == "__main__":
    plot_results(tensorboard_log= os.path.join('logs/PPO', 'FeedingSeparateRewardBaxter-v1-version9', 'lr_0.0003_batch_320_nenvs_16_nsteps_200_ent_0.0_hidden_512_sde_1_sdefreq_4_targetkl_0.03_gae_0.98_clip_0.2_nepochs_10_robot_reward_seed_856'), 
                  total_timesteps=16000000,
                  n_envs=16,
                  n_steps=200,
                  env_name='FeedingSeparateRewardBaxter-v1')
    # path_list = [os.path.join('logs/PPO', 'FeedingSeparateRewardBaxter-v1-version9', 'lr_0.0003_batch_320_nenvs_16_nsteps_200_ent_0.0_hidden_512_sde_1_sdefreq_4_targetkl_0.03_gae_0.98_clip_0.2_nepochs_10_total_reward_seed_856'),
    #               os.path.join('logs/PPO', 'FeedingSeparateRewardBaxter-v1-version9', 'lr_0.0003_batch_320_nenvs_16_nsteps_200_ent_0.0_hidden_512_sde_1_sdefreq_4_targetkl_0.03_gae_0.98_clip_0.2_nepochs_10_robot_reward_seed_856')]
    # plot_results_compare(tensorboard_log_list= path_list, 
    #               total_timesteps=16000000,
    #               n_envs=16,
    #               n_steps=200,
    #               env_name='FeedingSeparateRewardBaxter-v1')
