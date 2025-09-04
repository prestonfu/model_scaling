import re
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from qscaled.wandb_utils import flatten_dict, get_wandb_run_history
from qscaled.wandb_utils.multiple_seeds_per_run import MultipleSeedsPerRunCollector

from scripts.constants import DEFAULT_MAX_STEPS


class SimbaCollector(MultipleSeedsPerRunCollector):
    def __init__(
        self,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: List[str] | str = [],
        use_cached: bool = True,
        parallel: bool = True,
    ):
        super().__init__(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
            use_cached=use_cached,
            parallel=parallel,
        )

    def _set_hparams(self):
        self._hparams = [
            'env',
            'critic_width',
            'utd',
            'batch_size',
        ]

    def _set_wandb_metrics(self):
        self._wandb_metrics = [
            'avg_return',
            'critic/loss',
            'val_critic/loss',
        ]

    def _generate_key(self, run):
        config = flatten_dict(run.config)
        env = config['env.env_name']
        critic_width = config['agent.critic_hidden_dim']
        utd = config['updates_per_interaction_step']
        batch_size = config['buffer.sample_batch_size']

        key = (
            env,
            critic_width,
            utd,
            batch_size,
        )  # key is given in same order as `self._hparams`
        return key

    def wandb_fetch(self, run) -> Union[Tuple[Dict[str, Any], pd.DataFrame], None]:
        """Returns run metadata and history. If fails, returns None."""

        config = flatten_dict(run.config)
        num_seeds = 1  # not actually 1, but unfortunately that's all simba logging supports

        # fetch all data, then select the correct columns; else missing data
        result = get_wandb_run_history(run)
        if result is None or len(result) == 0:
            return None

        df = result
        last_step = df[self._env_step_key].iloc[-1] if len(df) > 0 else 0

        keys = [self._env_step_key] + self._wandb_metrics

        df[df == 'NaN'] = np.nan
        df = df.reindex(columns=keys, fill_value=np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')

        df.columns = [self._env_step_key] + [
            f'seed0/{k.replace("/", "_")}' for k in self._wandb_metrics for i in range(num_seeds)
        ]

        metadata = {
            'id': run.id,
            'name': run.name,
            'group': run.group,
            'host': run.metadata['host'] if run.metadata else None,
            'state': run.state,
            'last_step': last_step,
            'num_seeds': num_seeds,
            'metadata': run.metadata,
            'config': run.config,
        }
        return metadata, df

    def _update_wandb_metrics(self):
        self._wandb_metrics = [k.replace('/', '_') for k in self._wandb_metrics]

    def _combine_metadatas(self, metadatas):
        """
        Combine metadata from multiple runs. This function is called when
        flattening the data.
        """
        combined_metadata = defaultdict(list)
        for metadata in metadatas:
            for key, value in metadata.items():
                combined_metadata[key].append(value)

        combined_metadata['num_seeds'] = sum(metadata['num_seeds'] for metadata in metadatas)

        combined_metadata['last_step'] = (
            sum(metadata['last_step'] * metadata['num_seeds'] for metadata in metadatas)
            / combined_metadata['num_seeds']
        )

        return combined_metadata

    def prepare_zip_export_data(self, round_logging_freq=None) -> Dict[Any, np.ndarray]:
        data_dict = {}
        flattened_collector = self.flatten(round_logging_freq=round_logging_freq)

        for metric in self._wandb_metrics:
            for env in self.get_unique('env'):
                for critic_width in self.get_unique('critic_width', filter_str=f'env=="{env}"'):
                    filtered_collector = flattened_collector.filter(
                        f'env=="{env}" and critic_width=={critic_width}'
                    )

                    for key, (metadatas, rundatas) in filtered_collector.items():
                        rundata = rundatas[0]
                        save_key = (
                            metric.replace('/', '_'),
                            env,
                            critic_width,
                            key[self.hparam_index['utd']],
                            key[self.hparam_index['batch_size']],
                        )
                        subset = [self._env_step_key] + [
                            col for col in rundata.columns if re.match(f'seed\d+/{metric}$', col)
                        ]
                        data_dict[save_key] = rundata[subset].to_numpy()

        return data_dict

    def get_all_configs(self, max_steps=DEFAULT_MAX_STEPS):
        all_configs = []
        all_seeds = []
        for metadatas in self._metadatas.values():
            for metadata in metadatas:
                config = flatten_dict(metadata['config'])
                env = config['env_id']
                if metadata['state'] != 'running' and metadata['last_step'] < max_steps[env]:
                    continue
                config['id'] = metadata['id']
                config['name'] = metadata['name']
                config['group'] = metadata['group']
                all_configs.append(config)
                all_seeds.append(metadata['num_seeds'])
        return all_configs, all_seeds

    def get_closest(self, filter_str: str, target: Dict[str, Any]) -> Dict[str, Any]:
        filtered_rundatas = self.flatten().get_filtered_rundatas(filter_str)
        best_run = None
        best_distance = float('inf')

        for key, rundatas in filtered_rundatas.items():
            distance = 0
            for hparam in target:
                idx = self.hparam_index[hparam]
                distance += (np.log(target[idx]) - np.log(key[idx])) ** 2
            if distance < best_distance:
                best_distance = distance
                best_run = rundatas[0]

        return best_run

    def remove_incomplete_bad_runs(self, return_key: str, stop_at_offline_return: Dict[str, int]):
        for key in self.keys():
            metadatas, rundatas = self._metadatas[key], self._rundatas[key]
            new_metadatas, new_rundatas = [], []
            for metadata, rundata in zip(metadatas, rundatas):
                if metadata['state'] == 'running':
                    continue
                if metadata['last_step'] < metadata['config']['max_steps']:
                    num_seeds = metadata['num_envs']
                    return_keys = [f'seed{i}/{return_key}' for i in range(num_seeds)]
                    returns = np.mean(rundata[return_keys].dropna().values, axis=1)
                    if returns.max() < stop_at_offline_return.get(
                        metadata['config']['env_id'], float('inf')
                    ):
                        new_metadatas.append(metadata)
                        new_rundatas.append(rundata)
            self._metadatas[key] = new_metadatas
            self._rundatas[key] = new_rundatas
