#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:59:52 2021

@author: wenminggong

train a policy for HRI in assistive-gym using Decoupled PEBBLE
"""


import gym
import assistive_gym

from decouple_task_and_preference_pebble import DecoupledPEBBLE
from stable_baselines3.sac import MlpPolicy
from make_vec_separate_reward_env import make_vec_separate_reward_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from show_episode_reward_callbacks import ShowEpisodeRewardCallback
from separate_reward_vec_normalize import SeparateRewardVecNormalize
from reward_model import RewardModel

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import argparse
import yaml
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from torch import nn


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule. or can be used for ppo's clip_range
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    print(sys.version)
    
    parser = argparse.ArgumentParser(description='train pebble agent in assistive-gym')
    parser.add_argument("--env", type=str, default="FeedingSeparateRewardBaxter-v1", help="environment name")
    parser.add_argument("--reward-flag", type=str, default='total_reward', help="reward set: total_reward or robot_reward or pref_reward")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="logs/DecoupledPEBBLE_test/", type=str)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=2021)
    parser.add_argument("--n-envs", help="# of parallel environments, pebble supports single environment", type=int, default=1)
    parser.add_argument("--lr", help="learning rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", help="set replay buffer size", type=int, default=1000000)
    parser.add_argument("--learning-starts", help="how many steps before learning starts", type=int, default=600)
    parser.add_argument("--tau", help="soft target update", type=float, default=0.005)
    parser.add_argument("--gamma", help="discount factor", type=float, default=0.99)
    parser.add_argument("--train-freq", help="Update the model every train_freq steps", type=int, default=10)
    parser.add_argument("--gradient-steps", help="How many gradient steps to do after each rollout", type=int, default=1)
    parser.add_argument("--total-timesteps", help="total timesteps", type=int, default=16000000)
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, default=512)
    parser.add_argument("--ent-coef", help="Entropy regularization coefficient", default='auto_0.1')
    parser.add_argument("--target-entropy", help="target entropy when learning ent_coef", default='auto')
    parser.add_argument("--target-update", help="update the target network every # gradient steps", type=int, default=1)
    parser.add_argument("--hidden-dim", help="dim of hidden features", type=int, default=512)
    parser.add_argument("--num-layer", help="# of layers", type=int, default=2)
    parser.add_argument("--use-sde", help="Whether to use generalized State Dependent Exploration", type=int, default=1)
    parser.add_argument("--sde-freq", help="Sample a new noise matrix every n steps", type=int, default=4)
    parser.add_argument("--normalize", help="Normalize observation", type=int, default=1)
    parser.add_argument("--act-fun", help="activate function", type=str, default='relu')

    # unsupervised pretrain
    parser.add_argument("--unsuper-step", help="# of steps for unsupervised learning", type=int, default=10000)
    parser.add_argument("--unsuper-n-epochs", help="# of steps for unsupervised learning", type=int, default=50)
    parser.add_argument("--unsuper-flag", help="use unsupervised pre-train or not", type=str, default='True')
    # reward learning
    parser.add_argument("--re-lr", help="Learning rate of reward fn", type=float, default=3e-4)
    parser.add_argument("--re-segment", help="Size of segment", type=int, default=200)
    parser.add_argument("--re-act", help="Last activation for reward fn", type=str, default='tanh')
    parser.add_argument("--re-num-interaction", help="how many env_interactions to learn reward once", type=int, default=6000)
    parser.add_argument("--re-batch", help="Batch size for query", type=int, default=100)
    parser.add_argument("--re-update", help="Gradient update of reward fn", type=int, default=50)
    parser.add_argument("--re-feed-type", help="query strategy, 0: uniform, 1: disagreement, 2: entropy", type=int, default=1)
    parser.add_argument("--re-large-batch", help="size of buffer for ensemble uncertainty", type=int, default=50)
    parser.add_argument("--re-max-feed", help="# of total feedback", type=int, default=50000)
    parser.add_argument("--teacher-beta", type=float, default=-1)
    parser.add_argument("--teacher-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-eps-mistake", type=float, default=0.0)
    parser.add_argument("--teacher-eps-skip", type=float, default=0.0)
    parser.add_argument("--teacher-eps-equal", type=float, default=0.0)

    # decouple task and preference
    parser.add_argument("--reward-decay-type", help="linear or nolinear", type=str, default='linear')
    parser.add_argument("--reward-decay-rate", type=float, default=10.0)
    parser.add_argument("--reward-rou", type=float, default=0.01)
    args = parser.parse_args()
    
    max_ep_len = 200
    # log name
    env_name = args.env
        
    if args.normalize == 1:
        args.tensorboard_log += 'normalized_' + env_name
    else:
        args.tensorboard_log += env_name
    
    args.tensorboard_log += '/teacher_' + str(args.teacher_beta)
    args.tensorboard_log += '_' + str(args.teacher_gamma)
    args.tensorboard_log += '_' + str(args.teacher_eps_mistake)
    args.tensorboard_log += '_' + str(args.teacher_eps_skip)
    args.tensorboard_log += '_' + str(args.teacher_eps_equal)
    
    args.tensorboard_log += '/lr_'+str(args.lr)
    '''
    args.tensorboard_log += '_tau_' + str(args.tau)
    args.tensorboard_log += '_gamma_' + str(args.gamma)
    args.tensorboard_log += '_train-freq_' + str(args.train_freq)
    args.tensorboard_log += '_gradient-steps_' + str(args.gradient_steps)
    args.tensorboard_log += '_total-steps_' + str(args.total_timesteps)
    args.tensorboard_log += '_batch_' + str(args.batch_size)
    args.tensorboard_log += '_nenvs_' + str(args.n_envs)
    args.tensorboard_log += '_ent_' + str(args.ent_coef)
    args.tensorboard_log += '_hidden_' + str(args.hidden_dim)
    args.tensorboard_log += '_sde_' + str(args.use_sde)
    args.tensorboard_log += '_sdefreq_' + str(args.sde_freq)
    args.tensorboard_log += '_target-update_' + str(args.target_update)
    args.tensorboard_log += '_actfun_' + args.act_fun
    '''
    args.tensorboard_log += '_reward_lr' + str(args.re_lr)
    args.tensorboard_log += '_seg' + str(args.re_segment)
    args.tensorboard_log += '_react' + str(args.re_act)
    args.tensorboard_log += '_inter' + str(args.re_num_interaction)
    args.tensorboard_log += '_type' + str(args.re_feed_type)
    args.tensorboard_log += '_large' + str(args.re_large_batch)
    args.tensorboard_log += '_rebatch' + str(args.re_batch)
    args.tensorboard_log += '_reupdate' + str(args.re_update)
    args.tensorboard_log += '_maxfeed_' + str(args.re_max_feed)
    if args.unsuper_flag == 'True':
        args.unsuper_flag = True
    else:
        args.unsuper_flag = False
    args.tensorboard_log += '_unsuper_' + str(args.unsuper_flag)
    args.tensorboard_log += '_unsuper_steps_' + str(args.unsuper_step)
    args.tensorboard_log += '_update_' + str(args.unsuper_n_epochs)
    args.tensorboard_log += '_' + args.reward_flag
    
    args.tensorboard_log += '/reward_decay_type_' + args.reward_decay_type
    args.tensorboard_log += '_reward_decay_rate_' + str(args.reward_decay_rate)
    args.tensorboard_log += '_reward_decay_rou_' + str(args.reward_rou)
    
    # get system current time
    tic = time.perf_counter()
    # set random seeds
    random.seed(args.seed)
    for seed in [random.randint(0,1000) for _ in range(1)]:
        print('-----------------new test start: (seed = %d)-------------------' % seed)
        cur_tensorboard_log = args.tensorboard_log
        cur_tensorboard_log += '_seed_' + str(seed) 
    
        # extra params
        if args.use_sde == 0:
            use_sde = False
        else:
            use_sde = True
        
        # linear schedule clip_range, a function
        lr_range = linear_schedule(args.lr)
        reward_decay_range = linear_schedule(args.reward_decay_rate)
        
        # create vec env
        env = make_vec_separate_reward_env(env_id=env_name,
                                n_envs=args.n_envs,
                                monitor_dir=cur_tensorboard_log,
                                seed=seed,
                                vec_env_cls=SubprocVecEnv)
        
        # instantiating the reward model
        reward_model = RewardModel(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            size_segment=args.re_segment,
            activation=args.re_act, 
            lr=args.re_lr,
            mb_size=args.re_batch, 
            capacity = args.re_max_feed + 100,
            teacher_beta=args.teacher_beta, 
            teacher_gamma=args.teacher_gamma, 
            teacher_eps_mistake=args.teacher_eps_mistake,
            teacher_eps_skip=args.teacher_eps_skip, 
            teacher_eps_equal=args.teacher_eps_equal,
            large_batch=args.re_large_batch)
        
        # normalize observations to mean 0 and std 1
        if args.normalize == 1:
            env = SeparateRewardVecNormalize(env)
        
        # network arch
        net_arch = dict(pi=[args.hidden_dim]*args.num_layer, 
                         qf=[args.hidden_dim]*args.num_layer)
        if args.act_fun == 'tanh':
            policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.Tanh)
        elif args.act_fun == 'relu':
            policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.ReLU)
        else:
            policy_kwargs = dict(net_arch=net_arch)
    
        # train model
        model = DecoupledPEBBLE(
            reward_model,
            MlpPolicy,
            env,
            learning_rate=lr_range,
            buffer_size=args.buffer_size,  # 1e6
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            target_update_interval=args.target_update,
            target_entropy=args.target_entropy,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_freq,
            tensorboard_log=cur_tensorboard_log,
            policy_kwargs=policy_kwargs,
            seed=seed,
            num_interaction=args.re_num_interaction, 
            feed_type=args.re_feed_type,
            re_update=args.re_update,
            re_large_batch=args.re_large_batch,
            max_feed=args.re_max_feed,
            size_segment=args.re_segment,
            max_ep_len=max_ep_len,
            unsuper_step=args.unsuper_step,
            unsuper_n_epochs=args.unsuper_n_epochs,
            reward_flag=args.reward_flag,
            reward_decay_type = args.reward_decay_type,
            reward_decay_rate = reward_decay_range,
            reward_rou = args.reward_rou)
    
        # save args
        with open(os.path.join(cur_tensorboard_log, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
            yaml.dump(ordered_args, f)
        
        callback = ShowEpisodeRewardCallback(reward_flag=args.reward_flag)
        model.learn(total_timesteps=args.total_timesteps,
                    unsuper_flag=args.unsuper_flag,
                    callback=callback)
    
    print('total time:', time.perf_counter() - tic)