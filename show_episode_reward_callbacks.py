#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 21:01:24 2021

@author: wenminggong

custom callbacks to print episode total reward in console
"""


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import get_monitor_files, load_results
from stable_baselines3.common.results_plotter import ts2xy, plot_results

import os
import json
import pandas as pd


class ShowEpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose: int = 0, reward_flag: str='total_reward'):
        # reward_flag = total_reward or robot_reward or pref_reward
        super().__init__(verbose)
        self.reward_flag = reward_flag
        
        
    def _on_training_start(self):
        print('======================training start======================')
        
        
    def _on_training_end(self):
        print('======================training end========================')
        
        
    def _on_rollout_start(self):
        # print('**************rollout start***********')
        pass
        
        
    def _on_rollout_end(self):
        # print('**************rollout end**************')
        '''
        monitor_logs_path_list = get_monitor_files(self.model.tensorboard_log)
        sum_episode_total_reward = 0
        sum_episode_robot_reward = 0
        sum_episode_pref_reward = 0
        for logs_path in monitor_logs_path_list:
            with open(logs_path, "rt") as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#"
                header = json.loads(first_line[1:])
                data_frame = pd.read_csv(file_handler, index_col=None)
                sum_episode_total_reward += data_frame.tail(1)["total_reward"].values[0]
                sum_episode_robot_reward += data_frame.tail(1)["robot_reward"].values[0]
                sum_episode_pref_reward += data_frame.tail(1)["pref_reward"].values[0]
        
        mean_episode_total_reward = sum_episode_total_reward / len(monitor_logs_path_list)
        mean_episode_robot_reward = sum_episode_robot_reward / len(monitor_logs_path_list)
        mean_episode_pref_reward = sum_episode_pref_reward / len(monitor_logs_path_list)
        print('current total return: %.3f, robot return: %.3f, pref return: %.3f' % (mean_episode_total_reward, mean_episode_robot_reward, mean_episode_pref_reward))
        '''
        print('current timesteps: [%d / 16000000]' %self.model.num_timesteps)
        if self.num_timesteps % 800000 == 0:
            print("-------------------------saving model--------------------------")
            self.model.save(os.path.join(self.model.tensorboard_log, 'model', 'timesteps_'+str(self.model.num_timesteps)+"_ppo_model"))
            print("-------------------------saving env----------------------------")
            if not os.path.exists(os.path.join(self.model.tensorboard_log, 'env')):
                os.mkdir(os.path.join(self.model.tensorboard_log, 'env'))
            self.model.env.save(os.path.join(self.model.tensorboard_log, 'env', 'timesteps_'+str(self.model.num_timesteps)+"_env"))
            
        
    def _on_step(self):
        pass