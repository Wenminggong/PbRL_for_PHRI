#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:58:00 2021

@author: wenminggong

VecNormalize wrapper for separate-reward VecEnv, 
only normalize observation, not for separate rewards [total_r, robot_r, pref_r].
"""


import pickle
from copy import deepcopy
from typing import Any, Dict, Union

import gym
import numpy as np

from stable_baselines3.common import utils
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class SeparateRewardVecNormalize(VecNormalize):
    """
    A moving average, normalizing wrapper for separate-reward vectorized environment.
    has support for saving/loading moving average,

    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    """
    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        VecNormalize.__init__(
            self, 
            venv,
            training,
            norm_obs,
            norm_reward,
            clip_obs,
            clip_reward,
            gamma,
            epsilon,)
        # SeparateRewardVecEnv return rewards=np.stack([total_reward, robot_reward, pref_reward], ...)
        self.ret_rms = RunningMeanStd(shape=(3,))
        
    
    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new;
        rewards are not scaled
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rewards

        if self.training:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)

        obs = self.normalize_obs(obs)
        
        '''
        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)
        '''
        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        # self.ret[dones] = 0
        return obs, rewards, dones, infos
    
    
    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        # why use self.ret to update mean and std? 
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)
        # self.ret_rms.update(reward) # direct update reward moving mean and std
        