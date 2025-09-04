from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np
import pandas as pd
from qscaled.utils.zip_handler import BaseZipHandler

if TYPE_CHECKING:
    from qscaled.utils.configs import BaseConfig


class ModelSizeZipHandler(BaseZipHandler):
    """Handles saving and loading wandb collector offline returns data to/from zip files."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def save_prezip(self):
        """Saves data to prezip folder using wandb collector."""
        collector = self._config.wandb_collector
        data_dict = collector.prepare_zip_export_data(self._config.logging_freq)

        for key, data in data_dict.items():
            (
                metric,
                env,
                critic_width,
                critic_params,
                utd,
                batch_size,
                learning_rate,
                hard_target_update,
                hard_target_update_interval,
                target_noise_kind,
                target_noise,
                use_separate_critic,
                separate_critic_width,
                separate_critic_params,
                n_step,
                use_nstep_replay_buffer,
            ) = key
            filename = (
                f'{self._config.name}/'
                f'w{critic_width}p{critic_params}/'
                f'utd_{utd}/'
                f'{env}/'
                f'{metric.replace("/", "🦕")}/'
                f'bs_{batch_size}'
                f'_lr_{learning_rate}'
                f'_hard_{hard_target_update}_{hard_target_update_interval}'
                f'_targnoise_{target_noise_kind}_{target_noise}'
                f'_sep_{use_separate_critic}_w{separate_critic_width}p{separate_critic_params}'
                f'_nstep_{n_step}'
                f'_usenstep_{use_nstep_replay_buffer}'
                '.npy'
            )
            full_path = os.path.join(self._prezip_path, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            np.save(full_path, data)

    def parse_filename(self, filename):
        """
        Parses previously-saved filenames.

        Example input:
          gym_sweep/utd_1/Ant-v4/episode.return/bs_128_lr_0.0001.npy
        Example output:
          ('Ant-v4', 1.0, 128, 0.0001)
        """

        def parse_w_p(s):
            """Parses 'w256p619208' into (256, 619208)"""
            i = s.find('p')
            return int(s[1:i]), int(s[i + 1 :])

        path = os.path.splitext(filename)[0]
        _, critic_size_param, utd_param, env_name, metric, params = path.split('/')
        metric = metric.replace('🦕', '/')
        critic_width, critic_params = parse_w_p(critic_size_param)
        utd = float(utd_param[len('utd_') :])

        # ['bs', 128, 'lr', 0.0001, 'hard', True, 1000, 'targnoise', 'gaussian', 0.5, 'sep', False, 'w256p619208', 'nstep', 2, 'nstep_replay', True]
        (
            _,
            batch_size,
            _,
            learning_rate,
            _,
            hard_target_update,
            hard_target_update_interval,
            _,
            target_noise_kind,
            target_noise,
            _,
            use_separate_critic,
            separate_critic_size_str,
            _,
            n_step,
            _,
            use_nstep_replay_buffer,
        ) = params.split('_')
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        hard_target_update = hard_target_update == 'True'
        hard_target_update_interval = int(hard_target_update_interval)
        target_noise = float(target_noise)
        use_separate_critic = use_separate_critic == 'True'
        separate_critic_width, separate_critic_params = parse_w_p(separate_critic_size_str)
        n_step = int(n_step)
        use_nstep_replay_buffer = use_nstep_replay_buffer == 'True'

        return metric, {
            'env_name': env_name,
            'critic_width': critic_width,
            'critic_params': critic_params,
            'utd': utd,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hard_target_update': hard_target_update,
            'hard_target_update_interval': hard_target_update_interval,
            'target_noise_kind': target_noise_kind,
            'target_noise': target_noise,
            'use_separate_critic': use_separate_critic,
            'separate_critic_width': separate_critic_width,
            'separate_critic_params': separate_critic_params,
            'n_step': n_step,
            'use_nstep_replay_buffer': use_nstep_replay_buffer,
        }

    def load_df_from_zip(self) -> pd.DataFrame:
        """Loads data from zip file to a DataFrame."""
        full_path = os.path.join(self._zip_path, f'{self._config.name}.zip')
        records = defaultdict(dict)
        metrics = set()

        with ZipFile(full_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.endswith('.npy') and not filename.startswith('__MACOSX'):
                    metric, config = self.parse_filename(filename)
                    config_hashable = tuple(config.items())
                    env_name = config['env_name']
                    metrics.add(metric)

                    with zip_ref.open(filename, 'r') as f:
                        arr = np.load(f, allow_pickle=True)
                    step_data = arr[:, 0]
                    run_data = arr[:, 1:]

                    if metric == self._config.returns_key:
                        if env_name in self._config.max_returns:
                            run_data *= 1000 / self._config.max_returns[env_name]

                    good_cols = ~np.isnan(run_data).all(axis=0)

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        records[config_hashable].update(
                            {
                                'training_step': step_data,
                                metric: run_data,
                                'mean_' + metric: np.nanmean(run_data, axis=1),
                                'std_' + metric: np.nanstd(run_data, axis=1)
                                / np.sqrt(len(good_cols)),
                            }
                        )

        if len(records) == 0:
            raise ValueError('No data found in zip file.')

        reformatted_records = []
        for config_hashable, data in records.items():
            config = dict(config_hashable)
            reformatted_records.append(config | data)

        columns = list(config.keys()) + ['training_step']
        for metric in metrics:
            metric = ModelSizeZipHandler._rename_wandb_metric(metric)
            columns.extend([metric, 'mean_' + metric, 'std_' + metric])

        return pd.DataFrame(reformatted_records, columns=columns)
