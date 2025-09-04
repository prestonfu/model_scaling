import gymnasium as gym
import numpy as np

import os
import pickle
from collections import deque, namedtuple
from abc import ABC, abstractmethod

Batch = namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'dones', 'next_observations', 'discount_steps'],
)


class AbstractReplayBuffer(ABC):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        num_seeds: int,
        gamma: float,
    ):
        self.observation_space = observation_space
        self.observation_dim = observation_space.shape[-1]
        self.action_dim = action_dim
        self.capacity = capacity
        self.num_seeds = num_seeds
        self.gamma = gamma

    @abstractmethod
    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        pass

    @abstractmethod
    def sample_parallel(self, batch_size: int) -> Batch:
        pass

    @abstractmethod
    def sample_parallel_multibatch(
        self, batch_size: int, num_batches: int, include_idx=False
    ) -> Batch:
        pass


class ParallelReplayBuffer(AbstractReplayBuffer):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        num_seeds: int,
        gamma: float,
    ):
        super().__init__(observation_space, action_dim, capacity, num_seeds, gamma)
        self.observations = np.empty(
            (num_seeds, capacity, self.observation_dim),
            dtype=observation_space.dtype,
        )
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_seeds, capacity), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity), dtype=np.float32)
        self.dones_float = np.empty((num_seeds, capacity), dtype=np.float32)
        self.next_observations = np.empty(
            (num_seeds, capacity, self.observation_dim),
            dtype=observation_space.dtype,
        )
        self.discount_steps = np.ones((num_seeds, capacity), dtype=np.int32)

        self.contains_data = np.zeros(capacity, dtype=np.bool_)
        self.size = 0
        self.insert_index = 0
        self.earliest_index = 0
        self.capacity = capacity

    def update_obs_stats(self):
        self.obs_means = self.observations[:, : self.size, :].mean(1, keepdims=True)
        self.obs_vars = self.observations[:, : self.size, :].var(1, keepdims=True).sqrt() + 1e-3

    def update_rew_stats(self):
        self.rew_means = self.rewards[:, : self.size].mean(1, keepdims=True)
        self.rew_vars = self.rewards[:, : self.size].var(1, keepdims=True).sqrt() + 1e-3

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation
        self.contains_data[self.insert_index] = True

        self.insert_index = (self.insert_index + 1) % self.capacity
        if self.contains_data[self.insert_index]:
            self.earliest_index = self.insert_index
        self.size = min(self.size + 1, self.capacity)

    def get_at_index(self, i) -> Batch:
        return Batch(
            observations=self.observations[:, i],
            actions=self.actions[:, i],
            rewards=self.rewards[:, i],
            masks=self.masks[:, i],
            dones=self.dones_float[:, i],
            next_observations=self.next_observations[:, i],
            discount_steps=self.discount_steps[:, i],
        )

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return self.get_at_index(indx)

    def sample_parallel_multibatch(
        self, batch_size: int, num_batches: int, include_idx=False
    ) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        batch = self.get_at_index(indxs)
        if include_idx:
            return batch, indxs
        return batch

    def get_multibatch_at_index(self, indx: np.ndarray) -> Batch:
        """Expands to UTD = 1"""
        return self.get_at_index(indx[None, :])

    def sample_earliest_multibatch(self, batch_size: int, sample_frac=None) -> Batch:
        """
        * If `sample_frac is None`, returns the last batch_size.
        * If `type(sample_frac) == float`, samples batch_size points from the last
          `sample_frac` fraction of the buffer.
        Only samples 1 batch and makes num_batches copies of it.
        """
        if sample_frac is None:
            front_size = batch_size
            indxs = np.arange(self.earliest_index, self.earliest_index + front_size) % self.capacity
        else:
            front_size = int(self.size * sample_frac)
            valid_indxs = (
                np.arange(self.earliest_index, self.earliest_index + front_size) % self.capacity
            )
            indxs = np.random.choice(valid_indxs, size=batch_size)
        return self.get_multibatch_at_index(indxs)

    def sample_latest_multibatch(self, batch_size: int, sample_frac=None) -> Batch:
        """
        * If `sample_frac is None`, returns the last batch_size.
        * If `type(sample_frac) == float`, samples batch_size points from the last
          `sample_frac` fraction of the buffer.
        Only samples 1 batch and makes num_batches copies of it.
        """
        if sample_frac is None:
            back_size = batch_size
            indxs = np.arange(self.insert_index - back_size, self.insert_index) % self.capacity
        else:
            back_size = int(self.size * sample_frac)
            valid_indxs = (
                np.arange(self.insert_index - back_size, self.insert_index) % self.capacity
            )
            indxs = np.random.choice(valid_indxs, size=batch_size)
        return self.get_multibatch_at_index(indxs)

    def sample_state(self, batch_size: int) -> np.ndarray:
        indx = np.random.randint(self.size, size=batch_size)
        return self.observations[:, indx]

    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i * chunk_size : (i + 1) * chunk_size],
                self.actions[:, i * chunk_size : (i + 1) * chunk_size],
                self.rewards[:, i * chunk_size : (i + 1) * chunk_size],
                self.masks[:, i * chunk_size : (i + 1) * chunk_size],
                self.dones_float[:, i * chunk_size : (i + 1) * chunk_size],
                self.next_observations[:, i * chunk_size : (i + 1) * chunk_size],
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump(
            (self.size, self.insert_index),
            open(os.path.join(save_dir, 'buffer_info'), 'wb'),
        )

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, 'rb'))

            (
                self.observations[:, i * chunk_size : (i + 1) * chunk_size],
                self.actions[:, i * chunk_size : (i + 1) * chunk_size],
                self.rewards[:, i * chunk_size : (i + 1) * chunk_size],
                self.masks[:, i * chunk_size : (i + 1) * chunk_size],
                self.dones_float[:, i * chunk_size : (i + 1) * chunk_size],
                self.next_observations[:, i * chunk_size : (i + 1) * chunk_size],
            ) = data_chunk
        self.size, self.insert_index = pickle.load(
            open(os.path.join(save_dir, 'buffer_info'), 'rb')
        )


class NStepParallelReplayBuffer(AbstractReplayBuffer):
    """Note: this may be buggy."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        num_seeds: int,
        n_step: int,
        gamma: float,
    ):
        super().__init__(observation_space, action_dim, capacity, num_seeds, gamma)
        self.capacity = capacity
        self.n_step = n_step
        self.nstep_buffer = deque(maxlen=n_step)
        self.observations = np.empty((num_seeds, capacity, self.observation_dim), dtype=np.float32)
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_seeds, capacity), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity), dtype=np.float32)
        self.dones_float = np.empty((num_seeds, capacity), dtype=np.float32)
        self.next_observations = np.empty(
            (num_seeds, capacity, self.observation_dim), dtype=np.float32
        )
        self.discount_steps = np.empty((num_seeds, capacity), dtype=np.int32)

        self.insert_index = 0
        self.size = 0
        self.full = False

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.nstep_buffer.append((observation, action, reward, mask, done_float, next_observation))
        if len(self.nstep_buffer) == self.n_step:
            reward, next_obs, done, discount_step = self._get_n_step_info()
            obs_, act, _, _, _, _ = self.nstep_buffer[0]
            self.observations[:, self.insert_index] = obs_
            self.actions[:, self.insert_index] = act
            self.rewards[:, self.insert_index] = reward
            self.masks[:, self.insert_index] = mask
            self.dones_float[:, self.insert_index] = done
            self.next_observations[:, self.insert_index] = next_obs
            self.discount_steps[:, self.insert_index] = discount_step

            self.insert_index = (self.insert_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            if self.insert_index == 0:
                self.full = True

    def get_indx(self, batch_size: int):
        if self.size + self.n_step < self.capacity + 1:
            indx = np.random.randint(self.size - self.n_step + 1, size=batch_size)
        else:
            indx = self.generate_excluded_random(size=batch_size)
        return indx

    def get_at_index(self, i) -> Batch:
        return Batch(
            observations=self.observations[:, i],
            actions=self.actions[:, i],
            rewards=self.rewards[:, i],
            masks=self.masks[:, i],
            dones=self.dones_float[:, i],
            next_observations=self.next_observations[:, i],
            discount_steps=self.discount_steps[:, i],
        )

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = self.get_indx(batch_size)
        return self.get_at_index(indx)

    def sample_parallel_multibatch(
        self, batch_size: int, num_batches: int, include_idx=False
    ) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        batch = self.get_at_index(indxs)
        if include_idx:
            return batch, indxs
        return batch

    def _get_n_step_info(self):
        _, _, rewards, _, dones, next_observations = self.nstep_buffer[-1]
        discount_step = np.zeros_like(dones) + self.n_step
        for _, _, reward, _, done, next_observation in reversed(list(self.nstep_buffer)[:-1]):
            rewards = reward + self.gamma * rewards * (1 - done)
            if done.any():
                discount_step = np.where(done, discount_step - 1, discount_step)
                next_observations = np.where(done[:, None], next_observation, next_observations)
                dones = np.where(done, done, dones)
        return rewards, next_observations, dones, discount_step

    def generate_excluded_random(self, size=1):
        valid_numbers = (
            np.arange(self.n_step + 6, self.capacity) + self.insert_index - 3
        ) % self.capacity
        return np.random.choice(valid_numbers, size=size)
