#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:58:54 2021

@author: wenminggong

train a policy for HRI in assistive-gym using ppo
"""


import gym
import assistive_gym

from separate_reward_ppo import SeparateRewardPPO
from stable_baselines3.ppo import MlpPolicy
from make_vec_separate_reward_env import make_vec_separate_reward_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from show_episode_reward_callbacks import ShowEpisodeRewardCallback
from separate_reward_vec_normalize import SeparateRewardVecNormalize

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
    
    parser = argparse.ArgumentParser(description='train ppo agent in assistive-gym')
    parser.add_argument("--env", type=str, default="FeedingSeparateRewardBaxter-v1", help="environment name")
    parser.add_argument("--reward-flag", type=str, default='total_reward', help="reward set: total_reward or robot_reward or pref_reward")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="logs/PPO_test/", type=str)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=2021)
    parser.add_argument("--n-envs", help="# of parallel environments", type=int, default=16)
    parser.add_argument("--n-steps", help="# of steps to run for each environment per update", type=int, default=200)
    parser.add_argument("--lr", help="learning rate", type=float, default=3e-4)
    parser.add_argument("--total-timesteps", help="total timesteps", type=int, default=16000000)
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, default=256)
    parser.add_argument("--ent-coef", help="coeff for entropy", type=float, default=0.0)
    parser.add_argument("--hidden-dim", help="dim of hidden features", type=int, default=512)
    parser.add_argument("--num-layer", help="# of layers", type=int, default=2)
    parser.add_argument("--use-sde", help="Whether to use generalized State Dependent Exploration", type=int, default=1)
    parser.add_argument("--sde-freq", help="Sample a new noise matrix every n steps", type=int, default=4)
    parser.add_argument("--target-kl", help="Limit the KL divergence between updates", type=float, default=0.03)
    parser.add_argument("--gae-lambda", help="Factor for trade-off of bias vs variance", type=float, default=0.98)
    parser.add_argument("--clip-init", help="Initial value of clipping", type=float, default=0.2)
    parser.add_argument("--n-epochs", help="Number of epoch when optimizing the surrogate loss", type=int, default=20)
    parser.add_argument("--normalize", help="Normalize observation", type=int, default=1)    
    args = parser.parse_args()
    
    # log name
    env_name = args.env
        
    if args.normalize == 1:
        args.tensorboard_log += 'normalized_' + env_name + '/lr_'+str(args.lr)
    else:
        args.tensorboard_log += env_name + '/lr_'+str(args.lr)
        
    args.tensorboard_log += '_batch_' + str(args.batch_size)
    args.tensorboard_log += '_nenvs_' + str(args.n_envs)
    args.tensorboard_log += '_nsteps_' + str(args.n_steps)
    args.tensorboard_log += '_ent_' + str(args.ent_coef)
    args.tensorboard_log += '_hidden_' + str(args.hidden_dim)
    args.tensorboard_log += '_sde_' + str(args.use_sde)
    args.tensorboard_log += '_sdefreq_' + str(args.sde_freq)
    args.tensorboard_log += '_targetkl_' + str(args.target_kl)
    args.tensorboard_log += '_gae_' + str(args.gae_lambda)
    args.tensorboard_log += '_clip_' + str(args.clip_init)
    args.tensorboard_log += '_nepochs_' + str(args.n_epochs)
    args.tensorboard_log += '_' + args.reward_flag
    
    tic = time.clock()
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
        clip_range = linear_schedule(args.clip_init)
        # clip_range = args.clip_init
        lr_range = linear_schedule(args.lr)
        
        # create vec env
        env = make_vec_separate_reward_env(env_id=env_name,
                                n_envs=args.n_envs,
                                monitor_dir=cur_tensorboard_log,
                                seed=seed,
                                vec_env_cls=SubprocVecEnv)
        
        # normalize observations to mean 0 and std 1
        if args.normalize == 1:
            env = SeparateRewardVecNormalize(env)
        
        # network arch
        net_arch = [dict(pi=[args.hidden_dim]*args.num_layer, 
                         vf=[args.hidden_dim]*args.num_layer)]
        #TODO: 默认的激活函数是tanh，可以改为relu
        policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.ReLU)
    
        # train model
        model = SeparateRewardPPO(
            MlpPolicy, 
            env,
            tensorboard_log=cur_tensorboard_log, 
            seed=seed, 
            learning_rate=lr_range,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            use_sde=use_sde,
            sde_sample_freq=args.sde_freq,
            target_kl=args.target_kl,
            gae_lambda=args.gae_lambda,
            clip_range=clip_range,
            n_epochs=args.n_epochs,
            verbose=0,
            reward_flag=args.reward_flag)
    
        # save args
        with open(os.path.join(cur_tensorboard_log, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
            yaml.dump(ordered_args, f)
        
        callback = ShowEpisodeRewardCallback(reward_flag=args.reward_flag)
        model.learn(total_timesteps=args.total_timesteps,
                    callback=callback)
    
    print('total time:', time.clock() - tic)
    
    
