#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:58:54 2021

@author: wenminggong

train a policy for HRI in assistive-gym using preference-based ppo
adapted from B-Pref: https://github.com/pokaxpoka/B_Pref/blob/main/train_PrefPPO.py
"""


import gym
import assistive_gym
import argparse
import yaml
import os
from torch import nn
import time
import random

from collections import OrderedDict
from separate_reward_prefppo import SeparateRewardPrefPPO
from stable_baselines3.ppo import MlpPolicy
from make_vec_separate_reward_env import make_vec_separate_reward_env
from show_episode_reward_callbacks import ShowEpisodeRewardCallback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from separate_reward_vec_normalize import SeparateRewardVecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from reward_model import RewardModel


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="FeedingSeparateRewardBaxter-v1", help="environment name")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="logs/PrefPPO/", type=str)
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
    parser.add_argument("--n-epochs", help="Number of epoch when optimizing the surrogate loss", type=int, default=10)
    parser.add_argument("--normalize", help="Normalization", type=int, default=1)
    parser.add_argument("--act-fun", help="activate function", type=str, default='relu') 
    # unsupervised pretrain
    parser.add_argument("--unsuper-step", help="# of steps for unsupervised learning", type=int, default=32000)
    parser.add_argument("--unsuper-n-epochs", help="# of steps for unsupervised learning", type=int, default=50)
    parser.add_argument("--unsuper-flag", help="use unsupervised pre-train or not", type=bool, default=False)
    # reward learning
    parser.add_argument("--re-lr", help="Learning rate of reward fn", type=float, default=3e-4)
    parser.add_argument("--re-segment", help="Size of segment", type=int, default=200)
    parser.add_argument("--re-act", help="Last activation for reward fn", type=str, default='tanh')
    parser.add_argument("--re-num-interaction", help="how many env_episode_interactions to learn reward once", type=int, default=16000)
    parser.add_argument("--re-batch", help="Batch size for query", type=int, default=100)
    parser.add_argument("--re-update", help="Gradient update of reward fn", type=int, default=50)
    parser.add_argument("--re-feed-type", help="query strategy, 0: uniform, 1: disagreement, 2: entropy", type=int, default=1)
    parser.add_argument("--re-large-batch", help="size of buffer for ensemble uncertainty", type=int, default=10)
    parser.add_argument("--re-max-feed", help="# of total feedback", type=int, default=10000)
    parser.add_argument("--teacher-beta", type=float, default=-1)
    parser.add_argument("--teacher-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-eps-mistake", type=float, default=0.0)
    parser.add_argument("--teacher-eps-skip", type=float, default=0.0)
    parser.add_argument("--teacher-eps-equal", type=float, default=0.0)
    parser.add_argument("--reward-flag", type=str, default="total_reward")
    args = parser.parse_args()
    

    max_ep_len = 200
    
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
    args.tensorboard_log += '_reward_lr' + str(args.re_lr)
    args.tensorboard_log += '_seg' + str(args.re_segment)
    args.tensorboard_log += '_react' + str(args.re_act)
    args.tensorboard_log += '_inter' + str(args.re_num_interaction)
    args.tensorboard_log += '_type' + str(args.re_feed_type)
    args.tensorboard_log += '_large' + str(args.re_large_batch)
    args.tensorboard_log += '_rebatch' + str(args.re_batch)
    args.tensorboard_log += '_reupdate' + str(args.re_update)
    # args.tensorboard_log += '_batch_' + str(args.batch_size)
    # args.tensorboard_log += '_nenvs_' + str(args.n_envs)
    # args.tensorboard_log += '_nsteps_' + str(args.n_steps)
    # args.tensorboard_log += '_ent_' + str(args.ent_coef)
    # args.tensorboard_log += '_hidden_' + str(args.hidden_dim)
    # args.tensorboard_log += '_sde_' + str(args.use_sde)
    # args.tensorboard_log += '_sdefreq_' + str(args.sde_freq)
    # args.tensorboard_log += '_targetkl_' + str(args.target_kl)
    # args.tensorboard_log += '_gae_' + str(args.gae_lambda)
    # args.tensorboard_log += '_clip_' + str(args.clip_init)
    # args.tensorboard_log += '_nepochs_' + str(args.n_epochs)
    # args.tensorboard_log += '_actfun_' + args.act_fun
    args.tensorboard_log += '_maxfeed_' + str(args.re_max_feed)
    args.tensorboard_log += '_unsuper_' + str(args.unsuper_flag)
    args.tensorboard_log += '_unsuper_steps_' + str(args.unsuper_step)
    args.tensorboard_log += '_update_' + str(args.unsuper_n_epochs)
    args.tensorboard_log += '_reward_flag_' + args.reward_flag
    
    # get system current time
    tic = time.perf_counter()
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
        
        clip_range = linear_schedule(args.clip_init)
        # clip_range = args.clip_init
        lr_range = linear_schedule(args.lr)
        
        # Parallel environments
        env = make_vec_separate_reward_env(
                args.env,     
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
        
        if args.normalize == 1:
            # normalize observation
            env = SeparateRewardVecNormalize(env)
        
        # network arch
        net_arch = [dict(pi=[args.hidden_dim]*args.num_layer, 
                         vf=[args.hidden_dim]*args.num_layer)]
        if args.act_fun == 'tanh':
            policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.Tanh)
        elif args.act_fun == 'relu':
            policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.ReLU)
        else:
            policy_kwargs = dict(net_arch=net_arch)
        
        # train model
        model = SeparateRewardPrefPPO(
            reward_model,
            MlpPolicy, env,
            tensorboard_log=cur_tensorboard_log, 
            seed=seed, 
            learning_rate=lr_range,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            use_sde=use_sde,
            sde_sample_freq=args.sde_freq,
            target_kl= args.target_kl,
            gae_lambda=args.gae_lambda,
            clip_range=clip_range,
            n_epochs=args.n_epochs,
            num_interaction=args.re_num_interaction,
            feed_type=args.re_feed_type,
            re_update=args.re_update,
            max_feed=args.re_max_feed, 
            unsuper_step=args.unsuper_step,
            unsuper_n_epochs=args.unsuper_n_epochs,
            size_segment=args.re_segment,
            max_ep_len=max_ep_len,
            verbose=0,
            reward_flag=args.reward_flag)
    
        # save args
        with open(os.path.join(cur_tensorboard_log, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
            yaml.dump(ordered_args, f)
            
        # model.learn(total_timesteps=args.total_timesteps, unsuper_flag=True)
        callback = ShowEpisodeRewardCallback()
        model.learn(total_timesteps=args.total_timesteps,
                    unsuper_flag=args.unsuper_flag,
                    callback=callback)
    
    print('total time:', time.perf_counter() - tic)