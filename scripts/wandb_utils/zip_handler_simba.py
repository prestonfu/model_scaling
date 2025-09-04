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


class SimbaZipHandler(BaseZipHandler):
    """Handles saving and loading wandb collector offline returns data to/from zip files."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self._special_char = '🦕'

    def save_prezip(self):
        """Saves data to prezip folder using wandb collector."""
        collector = self._config.wandb_collector
        collector._update_wandb_metrics()
        data_dict = collector.prepare_zip_export_data(self._config.logging_freq)

        for key, data in data_dict.items():
            (
                metric,
                env,
                critic_width,
                utd,
                batch_size,
            ) = key
            filename = (
                f'{self._config.name}/'
                f'w{critic_width}/'
                f'utd_{utd}/'
                f'{env.replace("/", self._special_char)}/'
                f'{metric}/'
                f'bs_{batch_size}'
                '.npy'
            )
            full_path = os.path.join(self._prezip_path, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            np.save(full_path, data)

    def parse_filename(self, filename):
        """
        Parses previously-saved filenames.

        Example input:
          simbav2_model_scaling/w512/utd_1/Ant-v4/episode.return/bs_128.npy
        Example output:
          ('Ant-v4', 512, 1.0, 128)
        """

        path = os.path.splitext(filename)[0]
        _, critic_width_param, utd_param, env_name, metric, params = path.split('/')
        critic_width = int(critic_width_param[len('w') :])
        env_name = env_name.replace(self._special_char, '/')
        utd = float(utd_param[len('utd_') :])

        (
            _,
            batch_size,
        ) = params.split('_')
        batch_size = int(batch_size)

        return metric, {
            'env_name': env_name,
            'critic_width': critic_width,
            'utd': utd,
            'batch_size': batch_size,
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
            metric = SimbaZipHandler._rename_wandb_metric(metric)
            columns.extend([metric, 'mean_' + metric, 'std_' + metric])

        return pd.DataFrame(reformatted_records, columns=columns)
