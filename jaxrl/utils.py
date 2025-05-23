import os
import numpy as np
import time
import requests
import wandb
import logging
import ml_collections
import tensorflow_probability.substrates.numpy as tfp
from collections import defaultdict
from datetime import datetime
from typing import Dict


def mute_warning():
    tfp.distributions.TransformedDistribution(
        tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity()
    )
    logger = logging.getLogger('root')

    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return 'check_types' not in record.getMessage()

    logger.addFilter(CheckTypesFilter())


def log_seeds_to_wandb(step, infos, suffix: str = ''):
    dict_to_log = {'timestep': step}
    for info_key in infos:
        values = infos[info_key]
        for seed, value in enumerate(values):
            dict_to_log[f'seed{seed}/{info_key}{suffix}'] = value
        dict_to_log[f'mean/{info_key}{suffix}'] = np.mean(values)
        dict_to_log[f'stderr/{info_key}{suffix}'] = np.std(values) / np.sqrt(len(values))
    wandb.log(dict_to_log, step=step)


def is_using_slurm():
    return 'SLURM_JOB_ID' in os.environ


def get_flag_dict(FLAGS):
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(FLAGS, k) for k in FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def init_wandb(FLAGS, config):
    group_name = run_name = FLAGS.env_name
    group_name += f'_w{config["width_critic"]}'
    group_name += f'_utd{config["updates_per_step"]}'
    group_name += f'_lr{config["critic_lr"]}'
    group_name += f'_b{config["batch_size"]}'

    if is_using_slurm():
        if 'SLURM_ARRAY_JOB_ID' in os.environ and 'SLURM_ARRAY_TASK_ID' in os.environ:
            run_name += f'_{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        else:
            run_name += f'_{os.environ["SLURM_JOB_ID"]}'
        run_name += f'_sd{FLAGS.seed}'
    else:
        run_name += f'_sd{FLAGS.seed}'
    run_name += datetime.now().strftime('_%y%m%d-%H%M%S-%f')

    wandb.init(
        config=get_flag_dict(FLAGS),
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
        group=group_name,
        name=run_name,
        id=run_name,
        tags=[FLAGS.wandb_tag] if FLAGS.wandb_tag else ['debug'],
    )

    return run_name


def prefix_metrics(metrics, prefix, sep='/'):
    return {prefix + sep + key: value for key, value in metrics.items()}


class Timer:
    times = defaultdict(list)

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        Timer.times[self.key].append(elapsed)

    @classmethod
    def get_totals(cls):
        return {key: sum(times) for key, times in cls.times.items()}

    @classmethod
    def reset(cls):
        cls.times.clear()


class RollingMeter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = {}

    def add(self, step, key, value):
        if key not in self.data:
            self.data[key] = {'values': [], 'sum': 0.0, 'sum_sq': 0.0, 'count': 0}

        key_data = self.data[key]
        values = key_data['values']

        key_data['sum'] += value
        key_data['sum_sq'] += value**2
        key_data['count'] += 1
        values.append((step, value))

        while values and step - values[0][0] >= self.window_size:
            _, old_value = values.pop(0)
            key_data['sum'] -= old_value
            key_data['sum_sq'] -= old_value**2
            key_data['count'] -= 1

    def add_dict(self, step, input_dict: Dict):
        for key, value in input_dict.items():
            self.add(step, key, value)

    def mean(self, key):
        key_data = self.data.get(key)
        if key_data and key_data['count'] > 0:
            return key_data['sum'] / key_data['count']
        return None

    def std(self, key):
        key_data = self.data.get(key, None)
        if key_data and key_data['count'] > 1:
            mean = key_data['sum'] / key_data['count']
            variance = (key_data['sum_sq'] / key_data['count']) - mean**2
            return variance**0.5
        return None

    def keys(self):
        return self.data.keys()

    def get_log_data(self):
        log_data = {}
        for key in self.keys():
            data = self.mean(key)
            if data is not None:
                log_data[key] = data
        return log_data
