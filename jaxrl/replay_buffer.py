import gymnasium as gym
import numpy as np

import os
import pickle
import collections

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'dones', 'next_observations'],
)


class ParallelReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        num_seeds: int,
    ):
        self.observations = np.empty(
            (num_seeds, capacity, observation_space.shape[-1]),
            dtype=observation_space.dtype,
        )
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )
        self.masks = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )
        self.dones_float = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )
        self.next_observations = np.empty(
            (num_seeds, capacity, observation_space.shape[-1]),
            dtype=observation_space.dtype,
        )
        self.contains_data = np.zeros(capacity, dtype=np.bool_)
        self.size = 0
        self.insert_index = 0
        self.earliest_index = 0
        self.capacity = capacity
        self.n_parts = 4

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

    def sample_earliest_multibatch(
        self, batch_size: int, num_batches: int, sample_frac=None
    ) -> Batch:
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
        indxs = np.repeat(indxs[None, :], num_batches, axis=0)
        return self.get_at_index(indxs)

    def sample_latest_multibatch(
        self, batch_size: int, num_batches: int, sample_frac=None
    ) -> Batch:
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
        indxs = np.repeat(indxs[None, :], num_batches, axis=0)
        return self.get_at_index(indxs)

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
