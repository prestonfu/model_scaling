import re
from functools import reduce
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from qscaled.wandb_utils import flatten_dict, get_dict_value, get_wandb_run_history
from qscaled.wandb_utils.multiple_seeds_per_run import MultipleSeedsPerRunCollector

from scripts.generate_runs.constants import DEFAULT_MAX_STEPS


class BROModelScalingCollector(MultipleSeedsPerRunCollector):
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
            'benchmark',
            'env',
            'critic_width',
            'critic_params',
            'utd',
            'batch_size',
            'learning_rate',
            'hard_target_update',
            'hard_target_update_interval',
            'target_noise_kind',
            'target_noise',
            'use_separate_critic',
            'separate_critic_width',
            'separate_critic_params',
        ]

    def _set_wandb_metrics(self):
        self._wandb_metrics = [
            'return',
            'critic_loss',
            'new_data_critic_loss',
            'old_data_critic_loss',
            'validation_critic_loss',
            'critic_pnorm',
            'critic_gnorm',
            'critic_agnorm',
            'mean_critic_grad_var',
            'adam_mean_critic_grad_var',
            'relative_grad_var',
            'grad_norm_b64',
            'grad_norm_b128',
            'grad_norm_b256',
            'grad_norm_b512',
            'grad_norm_b1024',
            'old_target_critic_loss_1',
            'old_target_critic_loss_2',
            'old_target_critic_loss_4',
            'old_target_critic_loss_8',
            'old_target_critic_loss_16',
            'old_target_critic_loss_32',
            'old_target_critic_loss_64',
            'old_target_critic_loss_128',
            'old_target_critic_loss_256',
            'old_target_critic_loss_512',
            'gt_return_to_go_avg',
            'gt_return_to_go_0.0',
            'gt_return_to_go_0.25',
            'gt_return_to_go_0.5',
            'gt_return_to_go_0.75',
            'gt_return_to_go_1.0',
            'eval_q_avg',
            'eval_q_0.0',
            'eval_q_0.25',
            'eval_q_0.5',
            'eval_q_0.75',
            'eval_q_1.0',
            'gt_loss_avg',
            'gt_loss_0.0',
            'gt_loss_0.25',
            'gt_loss_0.5',
            'gt_loss_0.75',
            'gt_loss_1.0',
            'q1',
            'q2',
            'separate_q1',
            'separate_q2',
            'replay_separate_critic_loss',
            'separate_critic_loss',
            'new_data_separate_critic_loss',
            'old_data_separate_critic_loss',
            'sep_validation_critic_loss',
            'sep_grad_norm_b64',
            'sep_grad_norm_b128',
            'sep_grad_norm_b256',
            'sep_grad_norm_b512',
            'sep_grad_norm_b1024',
        ]

    def _generate_key(self, run):
        config = flatten_dict(run.config)
        benchmark = config['benchmark']
        env = config['env_name']
        critic_width = config['agent.width_critic']
        critic_params = config['agent.model_size.critic']
        utd = config['agent.updates_per_step']
        batch_size = config['agent.batch_size']
        learning_rate = config['agent.critic_lr']
        hard_target_update = config.get('agent.hard_target_update', False)
        hard_target_update_interval = config.get('agent.hard_target_update_interval', 1000)
        target_noise_kind = config.get('agent.target_noise_kind', 'none')
        target_noise = config.get('agent.target_noise', 0)
        use_separate_critic = config.get('agent.use_separate_critic', False)
        separate_critic_width = config.get('agent.separate_width_critic', 512)
        separate_critic_params = config.get('agent.model_size.separate_critic', 1)
        key = (
            benchmark,
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
        )  # key is given in same order as `self._hparams`
        return key

    def wandb_fetch(self, run) -> Union[Tuple[Dict[str, Any], pd.DataFrame], None]:
        """Returns run metadata and history. If fails, returns None."""
        config = flatten_dict(run.config)
        num_seeds = config['num_seeds']

        # fetch all data, then select the correct columns; else missing data
        result = get_wandb_run_history(run)
        if result is None:
            return None

        df = result
        last_step = df[self._env_step_key].iloc[-1] if len(df) > 0 else 0
        if last_step < 1e5:
            return None

        # correct column names
        df = df.rename(
            columns={
                col: col.replace('adammean_critic_grad_var', 'adam_mean_critic_grad_var')
                for col in df.columns
            }
        )

        keys = [self._env_step_key] + [
            f'seed{i}/{k}' for k in self._wandb_metrics for i in range(num_seeds)
        ]

        df[df == 'NaN'] = np.nan
        df = df.reindex(columns=keys, fill_value=np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')

        metadata = {
            'id': run.id,
            'name': run.name,
            'group': run.group,
            'host': run.metadata['host'] if run.metadata else None,
            'state': run.state,
            'runtime_mins': float(run.summary['_runtime'] / 60),
            'last_step': last_step,
            'num_seeds': config['num_seeds'],
            'metadata': run.metadata,
            'config': run.config,
        }
        return metadata, df

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
        combined_metadata['runtime_mins'] = (
            sum(metadata['runtime_mins'] * metadata['num_seeds'] for metadata in metadatas)
            / combined_metadata['num_seeds']
        )
        combined_metadata['last_step'] = (
            sum(metadata['last_step'] * metadata['num_seeds'] for metadata in metadatas)
            / combined_metadata['num_seeds']
        )

        return combined_metadata

    def prepare_zip_export_data(self, logging_freq=None) -> Dict[Any, np.ndarray]:
        data_dict = {}
        flattened_collector = self.flatten(logging_freq=logging_freq)

        for metric in self._wandb_metrics:
            for env in self.get_unique('env'):
                for critic_width in self.get_unique('critic_width', filter_str=f'env=="{env}"'):
                    for utd in self.get_unique(
                        'utd',
                        filter_str=f'env=="{env}" and critic_width=={critic_width}',
                    ):
                        filtered_collector = flattened_collector.filter(
                            f'env=="{env}" and critic_width=={critic_width} and utd=={utd}'
                        )

                        for key, (metadatas, rundatas) in filtered_collector.items():
                            rundata = rundatas[0]
                            save_key = (
                                metric,
                                env,
                                critic_width,
                                key[self.hparam_index['critic_params']],
                                utd,
                                key[self.hparam_index['batch_size']],
                                key[self.hparam_index['learning_rate']],
                                key[self.hparam_index['hard_target_update']],
                                key[self.hparam_index['hard_target_update_interval']],
                                key[self.hparam_index['target_noise_kind']],
                                key[self.hparam_index['target_noise']],
                                key[self.hparam_index['use_separate_critic']],
                                key[self.hparam_index['separate_critic_width']],
                                key[self.hparam_index['separate_critic_params']],
                            )
                            subset = [self._env_step_key] + [
                                col
                                for col in rundata.columns
                                if re.match(f'seed\d+/{metric}$', col)
                            ]
                            data_dict[save_key] = rundata[subset].to_numpy()

        return data_dict

    def get_all_configs(self, max_steps=DEFAULT_MAX_STEPS):
        all_configs = []
        all_seeds = []
        for metadatas in self._metadatas.values():
            for metadata in metadatas:
                config = flatten_dict(metadata['config'])
                env = config['env_name']
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
                    num_seeds = metadata['num_seeds']
                    return_keys = [f'seed{i}/{return_key}' for i in range(num_seeds)]
                    returns = np.mean(rundata[return_keys].dropna().values, axis=1)
                    if returns.max() < stop_at_offline_return.get(
                        metadata['config']['env_name'], float('inf')
                    ):
                        new_metadatas.append(metadata)
                        new_rundatas.append(rundata)
            self._metadatas[key] = new_metadatas
            self._rundatas[key] = new_rundatas
