#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 10:24:36 2021

@author: wenminggong

off-policy algorithm with reward learning for separate reward assistive-gym
"""


import io
import pathlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from new_buffers import EntReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from logger import Logger
import utils


class SeparateRewardOffPolicyRewardAlgorithm(OffPolicyAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3) with reward learning for separate reward env
    :param reward model
    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    """

    def __init__(
        self,
        reward_model,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        _init_setup_model: bool = True,
        num_interaction: int = 5000, 
        feed_type: int = 0,
        re_update: int = 100,
        re_large_batch: int = 2,
        max_feed: int = 1400,
        size_segment: int = 25,
        max_ep_len: int = 1000,
        unsuper_step: int = 0,
        reward_flag: str = 'total_reward',
    ):

        super(SeparateRewardOffPolicyRewardAlgorithm, self).__init__(
            policy = policy,
            env = env,
            policy_base = policy_base,
            learning_rate = learning_rate,
            buffer_size = buffer_size,  # 1e6
            learning_starts = learning_starts,
            batch_size = batch_size,
            tau = tau,
            gamma = gamma,
            train_freq = train_freq,
            gradient_steps = gradient_steps,
            optimize_memory_usage = optimize_memory_usage,
            policy_kwargs = policy_kwargs,
            tensorboard_log = tensorboard_log,
            verbose = verbose,
            device = device,
            create_eval_env = create_eval_env,
            monitor_wrapper = monitor_wrapper,
            seed = seed,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            use_sde_at_warmup = use_sde_at_warmup,
        )
        
        # reward learning
        self.reward_model = reward_model
        self.thres_interaction = num_interaction
        self.feed_type = feed_type
        self.re_update = re_update
        self.traj_obsact = None
        self.traj_reward = None
        self.first_reward_train = 0
        self.num_interactions = 0
        self.max_feed = max_feed
        self.total_feed = 0
        self.labeled_feedback = 0
        self.noisy_feedback = 0 # bot be used
        self.reward_batch = self.reward_model.mb_size
        self.unsuper_step = unsuper_step
        self.avg_train_true_return = 0 
        self.size_segment = size_segment
        self.max_ep_len = max_ep_len
        
        self.custom_logger = Logger(tensorboard_log, save_tb=False, log_frequency=10000, agent='sac')
        
        self.reward_flag = reward_flag
        
        if _init_setup_model:
            self._setup_model()
            
            
    def _setup_model(self):
        super(SeparateRewardOffPolicyRewardAlgorithm, self)._setup_model()
        if self.unsuper_step > 0:
            self.unsuper_buffer = EntReplayBuffer(
                self.unsuper_step+100,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=False,
            )
            self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=self.device)
            
            
    def learn_reward(self) -> None:
        # update margin
        new_margin = np.mean(self.avg_train_true_return) * (self.size_segment / self.max_ep_len)
        self.reward_model.set_teacher_thres_skip(new_margin)
        self.reward_model.set_teacher_thres_equal(new_margin)
        
        if self.first_reward_train == 0:
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feed += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        # update reward
        for epoch in range(self.re_update):
            if self.reward_model.teacher_eps_equal > 0:
                train_acc = self.reward_model.train_soft_reward()
            else:
                train_acc = self.reward_model.train_reward()
            total_acc = np.mean(train_acc)
            
            if total_acc > 0.97:
                break;
                
        print("Reward function is updated!! ACC: " + str(total_acc))
        
        
    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.perf_counter() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        self.logger.record("reward/total_feed", self.total_feed)
        self.logger.record("reward/labeled_feedback", self.labeled_feedback)
        self.logger.record("reward/noisy_feedback", self.noisy_feedback)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.avg_train_true_return = safe_mean([ep_info["total_reward"] for ep_info in self.ep_info_buffer])
            self.logger.record("rollout/ep_total_reward_mean", safe_mean([ep_info["total_reward"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_robot_reward_mean", safe_mean([ep_info["robot_reward"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_pref_reward_mean", safe_mean([ep_info["pref_reward"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_task_success", safe_mean([ep_info["task_success"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        # if len(self.ep_success_buffer) > 0:
        #     self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)
    
    
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        store obs and reward directly

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # # Store only the unnormalized version
        # if self._vec_normalize_env is not None:
        #     new_obs_ = self._vec_normalize_env.get_original_obs()
        #     reward_ = self._vec_normalize_env.get_original_reward()
        # else:
        #     # Avoid changing the original ones
        #     self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_
        
        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                # action: original sampled action; buffer_action: scaled action
                obsact = np.concatenate((self._last_obs, action), axis=-1) # num_env x (obs+act)
                obsact = np.expand_dims(obsact, axis=1) # num_env x 1 x (obs+act) 

                # Rescale and perform action
                new_obs, reward, dones, infos = env.step(action)
                done = dones[0]
                # for separate reward, reward is an array [[total_reward, robot_reward, pref_reward]]
                if self.reward_flag == 'total_reward':
                    train_reward = reward[:, 0]
                elif self.reward_flag == 'robot_reward':
                    train_reward = reward[:, 1]
                else:
                    train_reward = reward[:, 2]
                batch_reward = train_reward.reshape(-1,1,1)
                # reward estimator
                pred_reward = self.reward_model.r_hat_batch(obsact)
                pred_reward = pred_reward.reshape(-1)
                
                if self.traj_obsact is None:
                    self.traj_obsact = obsact
                    self.traj_reward = batch_reward
                else:
                    self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                    self.traj_reward = np.concatenate((self.traj_reward, batch_reward), axis=1)
                
                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1
                self.num_interactions += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += train_reward[0]

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (original action and normalized observation)
                self._store_transition(replay_buffer, action, new_obs, pred_reward, dones, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                # add samples to buffer
                if self.total_feed < self.max_feed:
                    self.reward_model.add_data_batch(self.traj_obsact, self.traj_reward)
                # reset traj
                self.traj_obsact, self.traj_reward = None, None
                
                # train reward using random data
                if self.first_reward_train == 0:
                    self.learn_reward()
                    self.first_reward_train = 1
                    self.num_interactions = 0
                    # to compare with traditional RL, need reset num_timesteps
                    self.num_timesteps = 0
                else:
                    if self.num_interactions >= self.thres_interaction and self.total_feed < self.max_feed:
                        self.learn_reward()
                        self.num_interactions = 0
                
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_reward = []
                    ep_success = []
                    for idx, info in enumerate(infos):
                        maybe_ep_info = info.get("episode")
                        if maybe_ep_info is not None:
                            ep_reward.append(maybe_ep_info[self.reward_flag])
                            ep_success.append(maybe_ep_info['task_success'])
                    # ep_pred_reward = np.mean(np.sum(self.rollout_buffer.rewards, 0) + pred_reward)
                                
                    self.custom_logger.log('eval/episode_reward', np.mean(ep_reward), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_reward', np.mean(ep_reward), self.num_timesteps)
                    # if self.metaworld_flag:
                    #     self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.dump(self.num_timesteps)
                
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
    
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "SeparateRewardOffPolicyRewardAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
    
    
    def collect_rollouts_unsuper(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        unsuper_buffer: EntReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                # action: original sampled action; buffer_action: scaled action
                obsact = np.concatenate((self._last_obs, action), axis=-1) # num_env x (obs+act)
                obsact = np.expand_dims(obsact, axis=1) # num_env x 1 x (obs+act)
                
                obs_origin = env.get_original_obs()
                unsuper_buffer.add_obs(obs_origin)
                state_entropy = unsuper_buffer.compute_state_entropy(obs_origin)
                # print(state_entropy)
                self.s_ent_stats.update(state_entropy)
                # print(self.s_ent_stats.var)
                norm_state_entropy = state_entropy / self.s_ent_stats.std

                # Rescale and perform action
                new_obs, reward, dones, infos = env.step(action)
                done = dones[0]
                # for separate reward, reward is an array [[total_reward, robot_reward, pref_reward]]
                if self.reward_flag == 'total_reward':
                    train_reward = reward[:, 0]
                elif self.reward_flag == 'robot_reward':
                    train_reward = reward[:, 1]
                else:
                    train_reward = reward[:, 2]
                
                batch_reward = train_reward.reshape(-1,1,1)
                # reward estimator
                pred_reward = norm_state_entropy.reshape(-1).data.cpu().numpy()
                
                if self.traj_obsact is None:
                    self.traj_obsact = obsact
                    self.traj_reward = batch_reward
                else:
                    self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                    self.traj_reward = np.concatenate((self.traj_reward, batch_reward), axis=1)
                
                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += train_reward[0]

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (original action and normalized observation)
                self._store_transition(replay_buffer, action, new_obs, pred_reward, dones, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                # add samples to buffer
                if self.total_feed < self.max_feed:
                    self.reward_model.add_data_batch(self.traj_obsact, self.traj_reward)
                # reset traj
                self.traj_obsact, self.traj_reward = None, None
                '''
                # train reward using random data
                if self.first_reward_train == 0:
                    self.learn_reward()
                    self.first_reward_train = 1
                    self.num_interactions = 0
                    # to compare with traditional RL, need reset num_timesteps
                    self.num_timesteps = 0
                else:
                    if self.num_interactions >= self.thres_interaction and self.total_feed < self.max_feed:
                        self.learn_reward()
                        self.num_interactions = 0
                '''
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_reward = []
                    ep_success = []
                    for idx, info in enumerate(infos):
                        maybe_ep_info = info.get("episode")
                        if maybe_ep_info is not None:
                            ep_reward.append(maybe_ep_info[self.reward_flag])
                            ep_success.append(maybe_ep_info['task_success'])
                    # ep_pred_reward = np.mean(np.sum(self.rollout_buffer.rewards, 0) + pred_reward)
                                
                    self.custom_logger.log('eval/episode_reward', np.mean(ep_reward), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_reward', np.mean(ep_reward), self.num_timesteps)
                    # if self.metaworld_flag:
                    #     self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.dump(self.num_timesteps)
                
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
    
    
    def learn_unsuper(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "SeparateRewardOffPolicyRewardAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            if self.num_timesteps < self.unsuper_step:
                # sample for unsuper pre-train
                rollout = self.collect_rollouts_unsuper(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                unsuper_buffer = self.unsuper_buffer,
                )
            else:
                if self.first_reward_train == 0:
                    self.learn_reward()
                    self.num_interactions = 0
                    self.first_reward_train = 2
                    
                    # reset value_net weights
                    # TODO: need to reset value-net weight
                    self.policy.reset_critic()
                rollout = self.collect_rollouts(
                    self.env,
                    train_freq=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                if self.first_reward_train == 2:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                else:
                    self.train_unsuper(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    print('train_unsuper!')

        callback.on_training_end()

        return self
    