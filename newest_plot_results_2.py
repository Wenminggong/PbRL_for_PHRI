#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:04:51 2022

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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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
                total_return_array[i, :] += data_frame['total_reward'].values[:total_timesteps //  (n_envs * n_steps)]
                task_success_array[i, :] += data_frame['task_success'].values[:total_timesteps //  (n_envs * n_steps)]
                pref_return_array[i, :] += data_frame['pref_reward'].values[:total_timesteps //  (n_envs * n_steps)]
    total_return_array = total_return_array / n_envs
    task_success_array = task_success_array / n_envs
    pref_return_array = pref_return_array / n_envs
    
    # 对数据作平滑处理
    # sac:100, ppo:5
    smooth = 150
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


def plot_curve(axs, x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array, color):
    index = np.arange(0, len(x), len(x) // 100)
    axs[0].plot(x[index], mean_total_return_array[index], lw=4, color=color)
    # axs[0].fill_between(x[index], mean_total_return_array[index] - std_total_return_array[index], mean_total_return_array[index]+std_total_return_array[index], alpha=.3, lw=0, color=color)
    axs[0].set_ylabel('Episode True Return', fontsize='xx-large')
    axs[0].set_xlabel('Environment Steps', fontsize='xx-large')
    axs[0].set_ylim(bottom=-400,top=160)
    axs[1].plot(x[index], mean_task_success_array[index], lw=4, color=color)
    # axs[1].fill_between(x[index], mean_task_success_array[index] - std_task_success_array[index], mean_task_success_array[index]+std_task_success_array[index], alpha=.3, lw=0, color=color)
    axs[1].set_ylabel('Task Success Rate', fontsize='xx-large')
    axs[1].set_xlabel('Environment Steps', fontsize='xx-large')
    axs[2].plot(x[index], mean_pref_return_array[index], lw=4, color=color)
    # axs[2].fill_between(x[index], mean_pref_return_array[index] - std_pref_return_array[index], mean_pref_return_array[index]+std_pref_return_array[index], alpha=.3, lw=0, color=color)
    axs[2].set_ylabel('Episode Preference Return', fontsize='xx-large')
    axs[2].set_xlabel('Environment Steps', fontsize='xx-large')
    axs[2].set_ylim(bottom=-200, top=40)
    for i in range(3):
        axs[i].set_facecolor('white')
        axs[i].spines['bottom'].set_color('black')
        axs[i].spines['bottom'].set_linewidth('3.0')
        axs[i].spines['left'].set_color('black')
        axs[i].spines['left'].set_linewidth('3.0')
        axs[i].spines['top'].set_color('black')
        axs[i].spines['top'].set_linewidth('3.0')
        axs[i].spines['right'].set_color('black')
        axs[i].spines['right'].set_linewidth('3.0')
        axs[i].grid(color='black', axis='y', ls='--')
    


if __name__ == '__main__':
    # path_list = ['logs/Final_results/PrefPPO', 'logs/Final_results/robust_prefppo/prefppo/beta=1', 'logs/Final_results/robust_prefppo/prefppo/mistake=0.1', 
    #               'logs/Final_results/robust_prefppo/prefppo/skip=0.1', 'logs/Final_results/robust_prefppo/prefppo/equal=0.1', 'logs/Final_results/robust_prefppo/prefppo/gamma=0.99',
    #               'logs/Final_results/DecoupledPrefPPO', 'logs/Final_results/robust_prefppo/decoupled_prefppo/beta=1', 'logs/Final_results/robust_prefppo/decoupled_prefppo/mistake=0.1', 
    #               'logs/Final_results/robust_prefppo/decoupled_prefppo/skip=0.1', 'logs/Final_results/robust_prefppo/decoupled_prefppo/equal=0.1/optimal', 'logs/Final_results/robust_prefppo/decoupled_prefppo/gamma=0.99']
    # path_list = ['logs/Final_results/PEBBLE/teacher_-1_1.0_0.0_0.0_0.0', 'logs/Final_results/robust_pebble/pebble/beta=1', 'logs/Final_results/robust_pebble/pebble/mistake=0.1', 
    #               'logs/Final_results/robust_pebble/pebble/skip=0.1', 'logs/Final_results/robust_pebble/pebble/equal=0.1', 'logs/Final_results/robust_pebble/pebble/gamma=0.99',
    #               'logs/Final_results/DecoupledPEBBLE', 'logs/Final_results/robust_pebble/decoupled_pebble/beta=1', 'logs/Final_results/robust_pebble/decoupled_pebble/mistake=0.1', 
    #               'logs/Final_results/robust_pebble/decoupled_pebble/skip=0.1', 'logs/Final_results/robust_pebble/decoupled_pebble/equal=0.1', 'logs/Final_results/robust_pebble/decoupled_pebble/gamma=0.99']
    # path_list = ['logs/Final_results/DecoupledPrefPPO', 'logs/Final_results/lambda_decoupled_prefppo/linear_lambda_5', 'logs/Final_results/lambda_decoupled_prefppo/linear_lambda_10',
    #               'logs/Final_results/lambda_decoupled_prefppo/non_linear_lambda_5_rho_1e-5', 'logs/Final_results/lambda_decoupled_prefppo/non_linear_lambda_10_rho_1e-5', 'logs/Final_results/lambda_decoupled_prefppo/non_linear_lambda_30_rho_1e-5',
    #               'logs/Final_results/lambda_decoupled_prefppo/non_linear_lambda_5_rho_1e-6', 'logs/Final_results/lambda_decoupled_prefppo/non_linear_lambda_10_rho_1e-6', 'logs/Final_results/lambda_decoupled_prefppo/non_linear_lambda_30_rho_1e-6',]
    path_list = ['logs/Final_results/lambda_decoupled_pebble/linear_lambda_10', 'logs/Final_results/DecoupledPEBBLE', 'logs/Final_results/lambda_decoupled_pebble/linear_lambda_20', 
                  'logs/Final_results/lambda_decoupled_pebble/non_linear_lambda_5_rho_1e-5', 'logs/Final_results/lambda_decoupled_pebble/non_linear_lambda_10_rho_1e-5', 'logs/Final_results/lambda_decoupled_pebble/non_linear_lambda_30_rho_1e-5',
                  'logs/Final_results/lambda_decoupled_pebble/non_linear_lambda_5_rho_1e-6', 'logs/Final_results/lambda_decoupled_pebble/non_linear_lambda_10_rho_1e-6', 'logs/Final_results/lambda_decoupled_pebble/non_linear_lambda_30_rho_1e-6',]
    # PPO_total_timesteps 16000000; SAC_total_timesteps 8000000
    # total_timesteps = 16000000
    total_timesteps = 8000000
    # PPO_n_envs 16; SAC_n_envs 1
    # n_envs = 16
    n_envs = 1
    n_steps = 200
    color = ['red', 'gold', 'aqua', 'mediumseagreen', 'lightslategray', 'violet', 
             'indigo', 'brown', 'midnightblue', 'darkcyan', 'darkslategrey', 'chocolate']
    
    fig, axs = plt.subplots(1, 3, figsize=(30, 6))
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    for i in range(len(path_list)):
        file_path_list = os.listdir(path_list[i])
        x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array = read_csv_data(path_list[i], file_path_list, total_timesteps, n_envs, n_steps)
        plot_curve(axs, x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array, color[i])
    
    # fig.legend(labels=['Oracle PrefPPO', 'Noisy PrefPPO', 'Mistake PrefPPO', 'Skip PrefPPO', 'Equal PrefPPO', 'Myopic PrefPPO', 
    #                     'Oracle Decoupled PrefPPO', 'Noisy Decoupled PrefPPO', 'Mistake Decoupled PrefPPO', 'Skip Decoupled PrefPPO', 'Equal Decoupled PrefPPO', 'Myopic Decoupled PrefPPO'], 
    #             fontsize='large', loc='center right')
    # fig.legend(labels=['Oracle PEBBLE', 'Noisy PEBBLE', 'Mistake PEBBLE', 'Skip PEBBLE', 'Equal PEBBLE', 'Myopic PEBBLE', 
    #                     'Oracle Decoupled PEBBLE', 'Noisy Decoupled PEBBLE', 'Mistake Decoupled PEBBLE', 'Skip Decoupled PEBBLE', 'Equal Decoupled PEBBLE', 'Myopic Decoupled PEBBLE'], 
    #             fontsize='large', loc='center right')
    # fig.legend(labels=['Linear (lambda_0 = 1.0)', 'Linear (lambda_0 = 5.0)', 'Linear (lambda_0 = 10.0)',
    #                     'Non-linear (lambda_0 = 5.0, rho = 1e-5)','Non-linear (lambda_0 = 10.0, rho = 1e-5)','Non-linear (lambda_0 = 30.0, rho = 1e-5)',
    #                     'Non-linear (lambda_0 = 5.0, rho = 1e-6)','Non-linear (lambda_0 = 10.0, rho = 1e-6)','Non-linear (lambda_0 = 30.0, rho = 1e-6)',], 
    #             fontsize='large',  loc='upper left', bbox_to_anchor=(0.75, 0.55))
    fig.legend(labels=['Linear (lambda_0 = 10.0)', 'Linear (lambda_0 = 15.0)', 'Linear (lambda_0 = 20.0)',
                        'Non-linear (lambda_0 = 5.0, rho = 1e-5)','Non-linear (lambda_0 = 10.0, rho = 1e-5)','Non-linear (lambda_0 = 30.0, rho = 1e-5)',
                        'Non-linear (lambda_0 = 5.0, rho = 1e-6)','Non-linear (lambda_0 = 10.0, rho = 1e-6)','Non-linear (lambda_0 = 30.0, rho = 1e-6)',], 
                fontsize='large',  loc='upper left', bbox_to_anchor=(0.75, 0.55))
    plt.savefig('lambda_off_policy', format='pdf')
    plt.show()