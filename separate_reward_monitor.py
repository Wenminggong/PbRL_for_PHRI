#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:11:55 2021

@author: wenminggong

monitor wrapper for separate rewards
"""

import csv
import json
import os
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.monitor import Monitor, ResultsWriter


class SeparateRewardMonitor(Monitor):
    """
    A monitor wrapper for assistive-gym environments with separate rewards, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        super(SeparateRewardMonitor, self).__init__(env=env,
                                                    filename=filename,
                                                    allow_early_resets=allow_early_resets,
                                                    reset_keywords=reset_keywords,
                                                    info_keywords=info_keywords)
        self.t_start = time.time()
        if filename is not None:
            self.results_writer = SeparateRewardResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords,
            )
        else:
            self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets

        self.total_rewards = None
        self.robot_rewards = None
        self.pref_rewards = None
        self.needs_reset = True
        self.episode_total_returns = []
        self.episode_robot_returns = []
        self.episode_pref_returns = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()
        
        
    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        # if not self.allow_early_resets and not self.needs_reset:
        #     raise RuntimeError(
        #         "Tried to reset an environment before done. If you want to allow early resets, "
        #         "wrap your env with Monitor(env, path, allow_early_resets=True)"
        #     )
        self.rewards = []
        self.total_rewards = []
        self.robot_rewards = []
        self.pref_rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.total_rewards.append(reward[0])
        self.robot_rewards.append(reward[1])
        self.pref_rewards.append(reward[2])
        if done:
            # self.needs_reset = True
            ep_total_rew = sum(self.total_rewards)
            ep_robot_rew = sum(self.robot_rewards)
            ep_pref_rew = sum(self.pref_rewards)
            ep_len = len(self.total_rewards)
            ep_info = {"total_reward": round(ep_total_rew, 6), "robot_reward": round(ep_robot_rew, 6), "pref_reward": round(ep_pref_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), "task_success": info['task_success']}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_total_returns.append(ep_total_rew)
            self.episode_robot_returns.append(ep_robot_rew)
            self.episode_pref_returns.append(ep_pref_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info


    def get_episode_rewards(self) -> List[float]:
        """
        Returns the total rewards of all the episodes

        :return:
        """
        return self.episode_total_returns
    
    
    def get_episode_robot_rewards(self) -> List[float]:
        """
        Returns the robot rewards of all the episodes

        :return:
        """
        return self.episode_robot_returns
    
    
    def get_episode_pref_rewards(self) -> List[float]:
        """
        Returns the pref rewards of all the episodes

        :return:
        """
        return self.episode_pref_returns
    
    
class SeparateRewardResultsWriter(ResultsWriter):
    """
    A result writer that saves the data from the `SeparateRewardMonitor` class

    :param filename: the location to save a log file, can be None for no log
    :param header: the header dictionary object of the saved csv
    :param reset_keywords: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[Dict[str, Union[float, str]]] = None,
        extra_keys: Tuple[str, ...] = (),
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        self.file_handler = open(filename, "wt")
        self.file_handler.write("#%s\n" % json.dumps(header))
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("total_reward", "robot_reward", "pref_reward", "l", "t", "task_success") + extra_keys)
        self.logger.writeheader()
        self.file_handler.flush()
