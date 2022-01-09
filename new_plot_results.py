#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:30:51 2022

@author: wenminggong

plot results
"""


import os
import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns
from stable_baselines3.common.monitor import get_monitor_files


def read_csv_data(par_path, tensorboard_log_list, total_timesteps, n_envs, n_steps):
    nums = len(tensorboard_log_list)
    x = np.arange(0, total_timesteps, n_envs * n_steps)
    
    total_return_array = np.zeros((nums, total_timesteps //  (n_envs * n_steps)))
    task_success_array = np.zeros((nums, total_timesteps //  (n_envs * n_steps)))
    pref_return_array = np.zeros((nums, total_timesteps //  (n_envs * n_steps)))
    
    for i in range(nums):
        cur_path = os.path.join(par_path, tensorboard_log_list[i])
        monitor_logs_path_list = get_monitor_files(cur_path)
        for logs_path in monitor_logs_path_list:
            with open(logs_path, "rt") as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#"
                header = json.loads(first_line[1:])
                data_frame = pd.read_csv(file_handler, index_col=None)
                total_return_array[i, :] += data_frame['total_reward'].values
                task_success_array[i, :] += data_frame['task_success'].values
                pref_return_array[i, :] += data_frame['pref_reward'].values
    total_return_array = total_return_array / n_envs
    task_success_array = task_success_array / n_envs
    pref_return_array = pref_return_array / n_envs
    
    # 对数据作平滑处理
    # sac:100, ppo:5
    smooth = 100
    smooth_total_return_array = np.zeros(total_return_array.shape)
    smooth_pref_return_array = np.zeros(pref_return_array.shape)
    smooth_task_success_array = np.zeros(task_success_array.shape)
    for j in range(total_return_array.shape[0]):
        for k in range(total_return_array.shape[1]):
            smooth_total_return_array[j, k] = np.mean(total_return_array[j, max(0, k-smooth+1): k+1])
            smooth_pref_return_array[j, k] = np.mean(pref_return_array[j, max(0,k-smooth+1): k+1])
            smooth_task_success_array[j, k] = np.mean(task_success_array[j, max(0, k-smooth+1): k+1])
    
    mean_total_return_array = np.mean(smooth_total_return_array, axis=0)
    std_total_return_array = np.std(smooth_total_return_array, axis=0)
    mean_pref_return_array = np.mean(smooth_pref_return_array, axis=0)
    std_pref_return_array = np.std(smooth_pref_return_array, axis=0)
    mean_task_success_array = np.mean(smooth_task_success_array, axis=0)
    std_task_success_array = np.std(smooth_task_success_array, axis=0)
    return x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array


def plot_curve(axs, x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array):
    index = np.arange(0, len(x), len(x) // 100)
    axs[0].plot(x[index], mean_total_return_array[index], lw=4, color='r')
    axs[0].fill_between(x[index], mean_total_return_array[index] - std_total_return_array[index], mean_total_return_array[index]+std_total_return_array[index], alpha=.3, lw=0, color='r')
    axs[0].set_ylabel('Episode True Return')
    axs[0].set_xlabel('Environment Steps')
    axs[1].plot(x[index], mean_task_success_array[index], lw=4, color='r')
    axs[1].fill_between(x[index], mean_task_success_array[index] - std_task_success_array[index], mean_task_success_array[index]+std_task_success_array[index], alpha=.3, lw=0, color='r')
    axs[1].set_ylabel('Success Rate')
    axs[1].set_xlabel('Environment Steps')
    axs[2].plot(x[index], mean_pref_return_array[index], lw=4, color='r')
    axs[2].fill_between(x[index], mean_pref_return_array[index] - std_pref_return_array[index], mean_pref_return_array[index]+std_pref_return_array[index], alpha=.3, lw=0, color='r')
    axs[2].set_ylabel('Episode Preference Return')
    axs[2].set_xlabel('Environment Steps')
    for i in range(3):
        axs[i].set_facecolor('white')
        axs[i].spines['bottom'].set_color('black')
        axs[i].spines['bottom'].set_linewidth('2.0')
        axs[i].spines['left'].set_color('black')
        axs[i].spines['left'].set_linewidth('2.0')
        axs[i].spines['top'].set_color('black')
        axs[i].spines['top'].set_linewidth('2.0')
        axs[i].spines['right'].set_color('black')
        axs[i].spines['right'].set_linewidth('2.0')
        axs[i].grid(color='black', axis='y', ls='--')
    


if __name__ == '__main__':
    path_list = ['logs/Final_results/SAC_total_reward']
    total_timesteps = 8000000
    n_envs = 1
    n_steps = 200
    
    fig, axs = plt.subplots(1, 3, figsize=(30, 6))
    
    for i in range(len(path_list)):
        file_path_list = os.listdir(path_list[i])
        x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array = read_csv_data(path_list[i], file_path_list, total_timesteps, n_envs, n_steps)
        plot_curve(axs, x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array)
    
    fig.legend(labels=['SAC with true reward'], loc='lower center')
    plt.savefig('SAC_total_reward')
    plt.show()