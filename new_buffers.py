#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:30:30 2021

@author: wenminggong

copy from B-pref
"""
import warnings
from typing import Dict, Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
    
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class EntReplayBuffer(BaseBuffer):
    """
    ReplayBuffer for sotring history observations, used to compute state entropy in unsupervised pre-train setting (PEBBLE and B-pref)  
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        batch_size: int = 0,
    ):
        super(EntReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.n_envs = n_envs
        
        if psutil is not None:
            total_memory_usage = self.observations.nbytes 
            
            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
                
    def compute_state_entropy(self, obs, k=5):
        batch_size = 500
        if self.full:
            full_obs = th.as_tensor(self.observations, device=self.device)
        else:
            full_obs = th.as_tensor(self.observations[:self.pos], device=self.device)
        obs = th.as_tensor(obs, device=self.device)
        
        with th.no_grad():
            dists = []
            for idx in range(len(full_obs) // batch_size + 1):
                start = idx * batch_size
                end = (idx + 1) * batch_size
                dist = th.norm(
                    obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
                )
                dists.append(dist)

            dists = th.cat(dists, dim=1)
            knn_dists = th.kthvalue(dists, k=k + 1, dim=1).values
            state_entropy = knn_dists
        return state_entropy.unsqueeze(1)

    def add_obs(self, obs: np.ndarray) -> None:
        # Copy to avoid modification by reference
        obs_size = obs.shape[0]        
        next_index = self.pos + self.n_envs
        if next_index >= self.buffer_size:
            self.full = True
            maximum_index = self.buffer_size - self.pos
            self.observations[self.pos:] = np.array(obs[:maximum_index]).copy()
            remain = self.n_envs - (maximum_index)
            if remain > 0:
                self.observations[0:remain] = np.array(obs[maximum_index:]).copy()
            self.pos = remain
        else:
            self.observations[self.pos:next_index] = np.array(obs).copy()
            self.pos = next_index

            
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # not be used in PbRL
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))