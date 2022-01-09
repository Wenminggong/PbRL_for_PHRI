#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 20:42:37 2021

@author: wenminggong

new function for policy evaluation
only for not observation normalization
"""


import assistive_gym
import gym
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from separate_reward_ppo import SeparateRewardPPO
from separate_reward_sac import SeparateRewardSAC
from separate_reward_vec_normalize import SeparateRewardVecNormalize
from make_vec_separate_reward_env import make_vec_separate_reward_env
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os


def evaluate_policy_hri(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv, str],
    n_eval_episodes: int = 20,
    seed: int = 856,
    deterministic: bool = True,
    n_envs: int = 10,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float, float, float, float, float, float], Tuple[List[float], List[float], List[float], List[int]]]:

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
        venv = SubprocVecEnv([make_env(env, i) for i in range(n_envs)])
    else:
        venv = env
       
    episode_total_returns = []
    episode_robot_returns = []
    episode_pref_returns = []
    episode_lengths = []
    task_success = []
    episode_count = 0
    while episode_count < n_eval_episodes:
        observations = venv.reset()
        current_total_returns = np.zeros(n_envs)
        current_robot_returns = np.zeros(n_envs)
        current_pref_returns = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        # states = None
        dones = [False] * venv.num_envs
        while not all(dones):
            actions, states = model.predict(observations, deterministic=deterministic)
            observations, rewards, dones, infos = venv.step(actions)
            total_reward = rewards[:, 0]
            robot_reward = rewards[:, 1]
            pref_reward = rewards[:, 2]
            current_total_returns += total_reward
            current_robot_returns += robot_reward
            current_pref_returns += pref_reward
            current_lengths += 1
        for i in range(len(infos)):
            task_success.append(infos[i]['task_success'])
        episode_total_returns += list(current_total_returns)
        episode_robot_returns += list(current_robot_returns)
        episode_pref_returns += list(current_pref_returns)
        episode_lengths += list(current_lengths)
        episode_count += n_envs
        
    mean_total_return = np.mean(np.array(episode_total_returns))
    mean_robot_return = np.mean(np.array(episode_robot_returns))
    mean_pref_return = np.mean(np.array(episode_pref_returns))
    std_total_return = np.std(np.array(episode_total_returns))
    std_robot_return = np.std(np.array(episode_robot_returns))
    std_pref_return = np.std(np.array(episode_pref_returns))
    task_success_rate = sum(task_success) / len(task_success)
    
    # print(episode_total_returns)
    # print(episode_pref_returns)
    if return_episode_rewards:
        return episode_total_returns, episode_robot_returns, episode_pref_returns, episode_lengths
    return mean_total_return, mean_robot_return, mean_pref_return, std_total_return, std_robot_return, std_pref_return, task_success_rate


def normalize_obs(obs, nor_venv):
    return np.clip((obs - nor_venv.obs_rms.mean) / np.sqrt(nor_venv.obs_rms.var + nor_venv.epsilon), -nor_venv.clip_obs, nor_venv.clip_obs)


if __name__ == "__main__":
    env_name = 'FeedingSeparateRewardBaxter-v1'
    # model = SeparateRewardPPO.load(os.path.join('logs/PPO_results', 'normalize_obs_tanh', 
    #                                             'lr_0.0003_batch_256_nenvs_16_nsteps_200_ent_0.0_hidden_256_sde_1_sdefreq_4_targetkl_0.03_gae_0.95_clip_0.3_nepochs_10_actfun_tanh_robot_reward_seed_856',
    #                                             'model', 'timesteps_2400000_ppo_model.zip'))
    model = SeparateRewardSAC.load(os.path.join('logs/PEBBLE_results', 'model', 'timesteps_4800000_ppo_model.zip'))
    # model = SeparateRewardSAC.load(os.path.join('logs/SAC_results', 'normalized_FeedingSeparateRewardBaxter-v1',
    #                                             'lr_0.0003_tau_0.005_gamma_0.99_train-freq_10_gradient-steps_1_total-steps_6400000_batch_512_nenvs_1_ent_auto_0.1_hidden_512_sde_1_sdefreq_4_target-update_1_actfun_relu_robot_reward_seed_856', 
    #                                             'model', 'timesteps_6400000_ppo_model.zip'))
    
    venv = make_vec_separate_reward_env(env_id=env_name,
                                n_envs=10,
                                vec_env_cls=SubprocVecEnv,
                                seed = 856)
    # nor_venv = SeparateRewardVecNormalize.load(os.path.join('logs/PPO_results', 'normalize_obs_tanh', 
    #                                             'lr_0.0003_batch_256_nenvs_16_nsteps_200_ent_0.0_hidden_256_sde_1_sdefreq_4_targetkl_0.03_gae_0.95_clip_0.3_nepochs_10_actfun_tanh_robot_reward_seed_856',
    #                                             'env', 'timesteps_2400000_env'), venv)
    # nor_venv = SeparateRewardVecNormalize.load(os.path.join('logs/SAC_results', 'normalized_FeedingSeparateRewardBaxter-v1',
    #                                             'lr_0.0003_tau_0.005_gamma_0.99_train-freq_10_gradient-steps_1_total-steps_6400000_batch_512_nenvs_1_ent_auto_0.1_hidden_512_sde_1_sdefreq_4_target-update_1_actfun_relu_robot_reward_seed_856', 
    #                                             'env', 'timesteps_6400000_env'), venv)
    nor_venv = SeparateRewardVecNormalize.load(os.path.join('logs/PEBBLE_results', 'env', 'timesteps_4800000_env'), venv)
    
    mean_total_return, mean_robot_return, mean_pref_return, std_total_return, std_robot_return, std_pref_return, task_success_rate = evaluate_policy_hri(model, nor_venv, n_eval_episodes=20)
    print('mean_total_return: {}, mean_robot_return: {}, mean_pref_return: {}'.format(mean_total_return, mean_robot_return, mean_pref_return))
    print('std_total_return: {}, std_robot_return: {}, std_pref_return: {}'.format(std_total_return, std_robot_return, std_pref_return))
    print('task_success_rate: {}'.format(task_success_rate))
    
    
    # render the policy in env
    env = gym.make(env_name)
    episode_total_returns = []
    episode_robot_returns = []
    episode_pref_returns = []
    length_list = []
    # while True:
    for i in range(2):
        env.seed(100 * i)
        env.action_space.seed(100 * i)
        done = False
        env.render()
        observation = env.reset()
        observation = normalize_obs(observation, nor_venv)
        episode_total_return = 0
        episode_robot_return = 0
        episode_pref_return = 0
        length = 0
        while not done:
            length += 1
            action, _states = model.predict(observation, deterministic=True)
            # print(action)
            observation, reward, done, info = env.step(action)
            observation = normalize_obs(observation, nor_venv)
            reward = np.array(reward)
            episode_robot_return += reward[1]
            episode_pref_return += reward[2]
            episode_total_return += reward[0]
    
        episode_total_returns.append(episode_total_return)
        episode_robot_returns.append(episode_robot_return)
        episode_pref_returns.append(episode_pref_return)
        length_list.append(length)
        
    print("total returns:", episode_total_returns)
    print("robot returns:", episode_robot_returns)
    print("pref returns:", episode_pref_returns)
    print("episode length:", length_list)
    
    nor_venv.close()
    env.close()
    
