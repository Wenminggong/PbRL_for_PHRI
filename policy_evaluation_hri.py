#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 20:42:37 2021

@author: wenminggong

new function for policy evaluation
adapted from stable_baselines3.common.evaluation
only for not observation normalization
"""


import assistive_gym
import gym
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from separate_reward_ppo import SeparateRewardPPO
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os


def evaluate_policy_hri(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv, str],
    n_eval_episodes: int = 100,
    seed: int = 856,
    deterministic: bool = True,
    n_envs: int = 4,
    reward_flag: str = 'pref_reward',
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:

    def make_env(env_id, rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id)
            else:
                env = env_id()
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            return env
        return _init
    
    if not isinstance(env, VecEnv):
        envs = SubprocVecEnv([make_env(env, i) for i in range(n_envs)])

    episode_rewards = []
    episode_lengths = []
    episode_count = 0
    n_steps = 200
    while episode_count < n_eval_episodes:
        observations = envs.reset()
        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        # states = None
        for _ in range(n_steps):
            actions, states = model.predict(observations, deterministic=deterministic)
            observations, rewards, dones, infos = envs.step(actions)
            # print("rewards:", rewards)
            if reward_flag == 'total_reward':
                pred_rewards = rewards[:, 0] + rewards[:, 1]
            elif reward_flag == 'robot_reward':
                pred_rewards = rewards[:, 0]
            else:
                pred_rewards = rewards[:, 1]
            current_rewards += pred_rewards
            current_lengths += 1
        episode_rewards += list(current_rewards)
        episode_lengths += list(current_lengths)
        episode_count += n_envs
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


if __name__ == "__main__":
    env_name = 'FeedingSeparateRewardBaxter-v1'
    model = SeparateRewardPPO.load(os.path.join('logs/PPO_in_Assistive-Gym', 'FeedingSeparateRewardBaxter-v1', 
                                                'lr_0.0003_batch_256_nenvs_16_nsteps_200_ent_0.0_hidden_512_sde_1_sdefreq_4_targetkl_0.03_gae_0.98_clip_0.2_nepochs_20_robot_reward_seed_856',
                                                'model', 'timesteps_16000000_ppo_model.zip'))
    # model = SeparateRewardPPO.load('timesteps_10400000_ppo_model.zip')
    
    # mean_reward, std_reward = evaluate_policy_hri(model=model, env=env_name, return_episode_rewards=False)
    # print('mean reward: {}, std reward: {}'.format(mean_reward, std_reward))
    print(model)
    print(model.env)
    # render the policy in env
    env = gym.make(env_name)
    # env = gym.wrappers.NormalizeObservation(env)
    episode_total_returns = []
    episode_robot_returns = []
    episode_pref_returns = []
    length_list = []
    # while True:
    for i in range(4):
        env.seed(856)
        env.action_space.seed(856)
        done = False
        env.render()
        observation = env.reset()
        episode_total_return = 0
        episode_robot_return = 0
        episode_pref_return = 0
        length = 0
        while not done:
            length += 1
            action, _states = model.predict(observation, deterministic=True)
            print(action)
            observation, reward, done, info = env.step(action)
            reward = np.array(reward)
            episode_robot_return += reward[0]
            episode_pref_return += reward[1]
            episode_total_return = episode_total_return + reward[0] + reward[1]
    
        episode_total_returns.append(episode_total_return)
        episode_robot_returns.append(episode_robot_return)
        episode_pref_returns.append(episode_pref_return)
        length_list.append(length)
        
    print("total returns:", episode_total_returns)
    print("robot returns:", episode_robot_returns)
    print("pref returns:", episode_pref_returns)
    print("episode length:", length_list)
    
