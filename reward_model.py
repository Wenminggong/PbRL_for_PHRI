import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
# import tqdm
import copy
import scipy.stats as st
import os
import time

from scipy.stats import norm
import psutil

device = 'cuda'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    # 3层隐藏层
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def compute_smallest_dist(obs, full_obs):
    # not be used
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=10000, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0):
        '''
        Parameters
        ----------
        ds : TYPE, int
            the dimension of state.
        da : TYPE, int
            the dimension of action.
        ensemble_size : TYPE, int
            use how many reward_predictor to ensemble. The default is 3.
        lr : TYPE, float
            reward predictor learning rate. The default is 3e-4.
        mb_size : TYPE, int
            sample mb_size trajectory pairs to query. The default is 128.
        size_segment : TYPE, int
            trajectory's length. The default is 1.
        env_maker : TYPE, optional
            DESCRIPTION. The default is None.
        max_size : TYPE, int
            the max size to store unlabel trajectories. The default is 100.
        activation : TYPE, str
            reward model activate function. The default is 'tanh'.
        capacity : TYPE, int
            the capacity of reward buffer. The default is 5e5.
        large_batch : TYPE, int
            initial query buffer size = large_batch * mb_size, then select mb_size to query. The default is 1.
        label_margin : TYPE, float
            reward margin to simulation human label. The default is 0.0.
        teacher_beta : TYPE, optional
            DESCRIPTION. The default is -1.
        teacher_gamma : TYPE, optional
            DESCRIPTION. The default is 1.
        teacher_eps_mistake : TYPE, optional
            DESCRIPTION. The default is 0.
        teacher_eps_skip : TYPE, optional
            DESCRIPTION. The default is 0.
        teacher_eps_equal : TYPE, optional
            DESCRIPTION. The default is 0.
        '''
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        
        # print buffer memory
        mem_available = psutil.virtual_memory().available
        print('----------------------available memory: %f G' % (mem_available / 1e9))
        mem_buffer_seg = self.buffer_seg1.nbytes * 2
        mem_buffer_label = self.buffer_label.nbytes
        print('----------------------reward buffer memory: %f G' % ((mem_buffer_seg + mem_buffer_label) / 1e9))
        
        # construct the ensemble reward neural networks
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 56
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # simulation human teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        # set batch_size by alpha * batch_size
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        # set batch_size dierctly
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    # construct ensemble reward models
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        # add an episode information to self.inputs and self.targets
        # add an obs, action and reward every time
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = (len(self.inputs) == 0)
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            # terminal state
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                # queue, cover first episode inputs
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                
    def add_data_batch(self, obses, rewards):
        # obses = np.array(n_envs, episode_length, obs_dim+act_dim)
        # rewards = np.array(n_envs, episode_length, 1)
        num_env = obses.shape[0]
        for index in range(num_env):
            # not limit self.inputs size
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        # test currect reward model
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        # 产生随机排列的索引
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        # 向上取整
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    
    def get_queries(self, mb_size=20):
        # get mb_size trajectory pairs from self.inputs and elf.targets
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
        
        if self.size_segment < len_traj:
            sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
            r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
            sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
            r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1
    
            # Generate time index 
            time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(mb_size)])
            time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
            time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
            
            sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
            r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
            sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
            r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
             
        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        # put query resutls to self.buffer_seg and self.buffer_label
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        # simulation human teacher
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        disagree = disagree[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def entropy_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        entropy = entropy[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = (labels == -1)
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc