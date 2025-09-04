import os
import numpy as np
import pandas as pd
import warnings
import pickle as pkl
from copy import deepcopy
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
from zipfile import ZipFile

from qscaled.constants import QSCALED_PATH
from qscaled.utils.state import remove_with_prompt

# Define the benchmarks
dog_humanoid_tasks = {
    'dog-run',
    'dog-stand',
    'dog-walk',
    'dog-trot',
    # 'humanoid-stand',
    'humanoid-walk',
    'humanoid-run',
}

dmc_tasks = {
    'acrobot-swingup',
    'cheetah-run',
    'finger-turn',
    'fish-swim',
    'hopper-hop',
    'quadruped-run',
    'walker-run',
}

metaworld_tasks = {
    'assembly-v2-goal-observable',
    'basketball-v2-goal-observable',
    'button-press-v2-goal-observable',
    'coffee-pull-v2-goal-observable',
    'coffee-push-v2-goal-observable',
    'disassemble-v2-goal-observable',
    'hammer-v2-goal-observable',
    'hand-insert-v2-goal-observable',
    'lever-pull-v2-goal-observable',
    'pick-place-wall-v2-goal-observable',
    'push-back-v2-goal-observable',
    'push-v2-goal-observable',
    'reach-v2-goal-observable',
    'stick-pull-v2-goal-observable',
    'sweep-v2-goal-observable',
}

myosuite_tasks = {
    'myo-key-turn-hard',
    'myo-key-turn',
    'myo-obj-hold-hard',
    'myo-obj-hold',
    'myo-pen-twirl-hard',
    'myo-pen-twirl',
    'myo-pose-hard',
    'myo-pose',
    'myo-reach-hard',
    'myo-reach',
}

# Create a dictionary mapping each task to its benchmark
task_to_benchmark = {}

for task in dog_humanoid_tasks:
    task_to_benchmark[task] = 'dog_humanoid'

for task in dmc_tasks:
    task_to_benchmark[task] = 'dmc'

for task in metaworld_tasks:
    task_to_benchmark[task] = 'metaworld'

for task in myosuite_tasks:
    task_to_benchmark[task] = 'myosuite'

task_to_benchmark['humanoid-stand'] = 'mine_dmc'


def load_old_bro_data(zip_path):
    eval_every_steps = 25000
    bro_param_count = {
        'xs': 0.55e6,
        's': 1.05e6,
        'sm': 2.83e6,
        'm': 4.92e6,
        'l': 26.31e6,
    }
    bro_param_count = {k: int(v) for k, v in bro_param_count.items()}

    data_entries = []

    with ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.npy'):
                parts = file.split('/')

                if len(parts) >= 2:
                    utd_part, model_size_part = parts[-2].split('_')
                    assert utd_part.startswith('rr')
                    utd = int(utd_part[2:])
                    param_count = bro_param_count[model_size_part]

                    task = os.path.splitext(parts[-1])[0]
                    if task not in task_to_benchmark:
                        continue
                    benchmark = task_to_benchmark[task]

                    with zip_ref.open(file, 'r') as f:
                        data_array = np.load(f, allow_pickle=True)

                    if benchmark in ['metaworld', 'myosuite']:  # normalize returns
                        data_array *= 1000

                    data_entries.append(
                        (utd, model_size_part, param_count, benchmark, task, data_array)
                    )

    df = pd.DataFrame(
        data_entries,
        columns=[
            'utd',
            'model_size_desc',
            'model_size',
            'benchmark',
            'env_name',
            'return',
        ],
    )
    df['training_step'] = df['return'].apply(
        lambda x: np.arange(1, x.shape[0] + 1) * eval_every_steps
    )
    df['mean_return'] = df['return'].apply(lambda x: np.mean(x, axis=1))
    df['std_return'] = df['return'].apply(lambda x: np.std(x, axis=1) / np.sqrt(x.shape[1]))
    df['lr'] = 3e-4
    df['batch_size'] = 1024
    return df


def filter_out_resets(df, reset_freq, window=0):
    """
    Filter out sharp drops in performance data. Useful for preprocessing returns
    when there are agent resets.
    - df: DataFrame loaded from zip
    - reset_freq: Number of gradient steps between resets
    - window: Number of steps to look forward for performance drop
    """
    filtered_df = deepcopy(df)

    # Prefill with junk values to match type
    cols_to_update = ['training_step', 'return', 'mean_return', 'std_return']
    for col in cols_to_update:
        filtered_df[f'{col}_resetfilter'] = filtered_df[col]
    filtered_df['reset_steps'] = filtered_df['training_step']

    for i, row in df.iterrows():
        training_step = row['training_step']
        mean_returns = row['mean_return']
        mask = ~np.isnan(mean_returns)
        training_step = training_step[mask]
        mean_returns = mean_returns[mask]
        for col in cols_to_update:
            filtered_df.at[i, f'{col}_resetfilter'] = filtered_df.at[i, col][mask]
        reset_freq_envsteps = int(reset_freq / row['utd'])
        reset_envsteps = np.arange(
            reset_freq_envsteps, training_step.max() + 1, reset_freq_envsteps
        )
        if len(reset_envsteps) == 0:
            continue

        # Hacky approach: check both 'left' or 'right' for searchsorted to determine
        # whether eval logging is before/after reset
        # Should be 'right' starting commit bbcf0a59

        post_reset_idxs_left = np.searchsorted(training_step, reset_envsteps, side='left')
        post_reset_idxs_right = np.searchsorted(training_step, reset_envsteps, side='right')
        if post_reset_idxs_right[-1] == len(training_step):
            post_reset_idxs_right = post_reset_idxs_right[:-1]

        mean_returns_pre_reset_left = mean_returns[post_reset_idxs_left - 1]
        mean_returns_post_reset_left = mean_returns[post_reset_idxs_left]
        mean_returns_pre_reset_right = mean_returns[post_reset_idxs_right - 1]
        mean_returns_post_reset_right = mean_returns[post_reset_idxs_right]

        if np.sum(mean_returns_pre_reset_left - mean_returns_post_reset_left) > np.sum(
            mean_returns_pre_reset_right - mean_returns_post_reset_right
        ):
            post_reset_idxs = post_reset_idxs_left
        else:
            post_reset_idxs = post_reset_idxs_right

        # Account for steps before first reset
        padded_curr_reset_idxs = np.append(0, post_reset_idxs) + window
        # Account for steps after last reset
        padded_next_reset_idxs = np.append(post_reset_idxs, len(training_step))
        mean_returns_pre_reset = np.append(float('-inf'), mean_returns[post_reset_idxs - 1])

        monotone_idx = []
        for (
            mean_return_pre_reset,
            post_reset_idx,
            post_next_reset_idx,
        ) in zip(
            mean_returns_pre_reset,
            padded_curr_reset_idxs,
            padded_next_reset_idxs,
        ):
            recovery_mask = (
                mean_returns[post_reset_idx:post_next_reset_idx] >= mean_return_pre_reset
            )
            recovery_idxs = np.where(recovery_mask)[0] + post_reset_idx
            if len(recovery_idxs) > 0:
                min_recovery_idx = recovery_idxs[0]
                monotone_idx.append(np.arange(min_recovery_idx, post_next_reset_idx))

        monotone_idx = np.concatenate(monotone_idx)

        for col in cols_to_update:
            filtered_df.at[i, f'{col}_resetfilter'] = filtered_df.at[i, f'{col}_resetfilter'][
                monotone_idx
            ]
        filtered_df.at[i, 'reset_steps'] = reset_envsteps

    return filtered_df


def dropna(df):
    """Recalculate mean/sd (metrics are missing from some runs)"""
    df = df.copy()
    for col in df.columns:
        if 'mean_' + col not in df.columns:
            continue
        for i, row in df.iterrows():
            data = row[col]
            good_cols = (~np.isnan(data)).all(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                df.at[i, col] = data
                df.at[i, 'mean_' + col] = np.nanmean(data, axis=1)
                df.at[i, 'std_' + col] = np.nanstd(data, axis=1) / np.sqrt(len(good_cols))
    return df


def fill_isotonic_regression(df, training_step_key='training_step', mean_return_key='mean_return'):
    """Modifies in place"""
    iso_reg_results = []

    for _, row in df.iterrows():
        ir = IsotonicRegression(out_of_bounds='clip')
        x = row[training_step_key]
        y = row[mean_return_key]
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        ir.fit(x, y)
        y_iso = ir.predict(x)
        iso_reg_results.append(y_iso)

    df['return_isotonic'] = iso_reg_results
    df['final_return_isotonic'] = df['return_isotonic'].apply(lambda x: x[-1])


def compute_thresholds_per_env(df, mode='worst', threshold_query: str = ''):
    """
    The lowest isotonic return from the thresholds specified by threshold_query.
    If threshold_query is empty str, no filter is applied.

    Mode sets the max threshold:
    - 'worst': return the lowest final isotonic return across all settings per env
    - 'best_per_utd_sigma': for each env, utd, sigma, find the best final isotonic
      return. Then, return the lowest across all utd, sigma per env.
    """
    df_for_thresholds = df.query(threshold_query) if threshold_query else df
    if mode == 'worst':
        max_threshold_per_env = df_for_thresholds.groupby(['env_name'])[
            'final_return_isotonic'
        ].min()
    elif mode == 'best_per_utd_sigma':
        max_threshold_per_env = df_for_thresholds.groupby(['env_name', 'utd', 'critic_width'])[
            'final_return_isotonic'
        ].max()
        max_threshold_per_env = max_threshold_per_env.groupby('env_name').min()
    else:
        raise ValueError

    df['achieve_highest_threshold'] = (
        df['final_return_isotonic'] >= max_threshold_per_env[df['env_name']].values
    )

    return max_threshold_per_env


def bootstrap_crossings_per_env(
    df,
    threshold_query: str = '',
    manual_thresholds={},
    max_threshold_to_thresholds_fn=None,
    training_step_key='training_step',
    return_key='return',
    filename: str = '',
    use_cached=True,
):
    """
    Almost identical to bootstrap_crossings from qscaled repo, but now we use
    the lowest isotonic return from the thresholds specified by threshold_query.
    If threshold_query is empty str, no filter is applied.

    manual_thresholds is a dictionary that maps environment names to lists of
    thresholds. If provided, overrides threshold_query.

    max_threshold_to_thresholds_fn is a function that takes the max threshold
    and returns the thresholds to use for each environment.
    """
    assert threshold_query or manual_thresholds, (
        'Either threshold_query or manual_thresholds must be provided.'
    )
    assert filename != ''

    if manual_thresholds:
        max_threshold_per_env = pd.Series(manual_thresholds)
    else:
        max_threshold_per_env = compute_thresholds_per_env(df, threshold_query)
    thresholds_per_env = pd.DataFrame(
        {'env': max_threshold_per_env.index, 'thresholds': max_threshold_per_env}
    ).apply(lambda row: max_threshold_to_thresholds_fn(row['env'], row['thresholds']), axis=1)
    # thresholds_per_env = max_threshold_per_env.apply(max_threshold_to_thresholds_fn)
    df['max_threshold'] = df['env_name'].map(max_threshold_per_env)
    df['thresholds'] = df['env_name'].map(thresholds_per_env)

    crossings_array = []

    for _, row in df.iterrows():
        row_crossings = []
        if not isinstance(row['thresholds'], (list, np.ndarray)) and np.isnan(row['thresholds']):
            raise ValueError(f'Thresholds are not a list or array for {row["env_name"]}')
        for threshold in row['thresholds']:
            # Get crossing from isotonic regression
            crossing_idx = np.where(row['return_isotonic'] >= threshold)[0]
            row_crossings.append(
                row[training_step_key][crossing_idx[0]] if len(crossing_idx) > 0 else np.nan
            )
        crossings_array.append(row_crossings)

    df['crossings'] = crossings_array

    bootstrap_cache_file = os.path.join(QSCALED_PATH, 'bootstrap_results', f'{filename}.pkl')

    if use_cached and os.path.exists(bootstrap_cache_file):
        with open(bootstrap_cache_file, 'rb') as f:
            results = pkl.load(f)
            iso_reg = results['iso_reg']
            iso_reg_stds = results['iso_reg_stds']
            crossings = results['crossings']
            crossings_mean = results['crossings_mean']
            crossings_std = results['crossings_std']

    else:
        iso_reg = []
        iso_reg_stds = []
        crossings = []
        crossings_mean = []
        crossings_std = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', category=RuntimeWarning)

            for i, row in tqdm(df.iterrows(), total=len(df)):
                # Number of bootstrap samples. 100 seems enough for std to converge
                n_bootstrap = 100
                ir = IsotonicRegression(out_of_bounds='clip')
                x = row[training_step_key]

                y_iso_samples = []
                sample_crossings = []
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    sample_indices = np.random.randint(
                        0, row[return_key].shape[1], size=row[return_key].shape[1]
                    )
                    # Average the bootstrap samples
                    y = np.nanmean(row[return_key][:, sample_indices], axis=1)

                    # Fit isotonic regression on this bootstrap sample
                    ir.fit(x, y)
                    y_iso = ir.predict(x)
                    y_iso_samples.append(y_iso)

                    # For each bootstrap sample, find threshold crossings
                    sample_crossing = []
                    for threshold in row['thresholds']:
                        crossing_idx = np.where(y_iso >= threshold)[0]
                        crossing = (
                            row[training_step_key][crossing_idx[0]]
                            if len(crossing_idx) > 0
                            else np.nan
                        )
                        sample_crossing.append(crossing)
                    sample_crossings.append(sample_crossing)

                # Store mean prediction, crossing statistics, and isotonic std
                iso_reg.append(y_iso_samples)
                crossings.append(np.array(sample_crossings))
                crossings_mean.append(np.nanmean(sample_crossings, axis=0))
                crossings_std.append(np.nanstd(sample_crossings, axis=0))
                iso_reg_stds.append(np.std(y_iso_samples, axis=0))

            if any('Degrees of freedom <= 0 for slice' in str(warning.message) for warning in w):
                print(
                    'It is likely that some environments do not reach every performance threshold '
                    'for every UTD. This can cause the standard deviation to be zero. '
                    'Consider decreasing your thresholds in the config, and call `bootstrap_crossings` '
                    'with `use_cached=False`.',
                    UserWarning,
                )

        # Save results to cache
        results = {
            'iso_reg': iso_reg,
            'iso_reg_stds': iso_reg_stds,
            'crossings': crossings,
            'crossings_mean': crossings_mean,
            'crossings_std': crossings_std,
        }
        remove_with_prompt(bootstrap_cache_file)
        os.makedirs(os.path.dirname(bootstrap_cache_file), exist_ok=True)
        with open(bootstrap_cache_file, 'wb') as f:
            pkl.dump(results, f)

    df['return_isotonic_bootstrap'] = iso_reg
    df['crossings_bootstrap'] = crossings
    df['crossings_bootstrap_mean'] = crossings_mean
    df['crossings_std'] = crossings_std
    df['return_isotonic_std'] = iso_reg_stds

    mean_std = np.nanmean(np.array(crossings_std))
    print(f'Average standard deviation across all conditions: {mean_std:.2f}')
    return df, max_threshold_per_env, thresholds_per_env


def truncate(df, max_steps):
    assert isinstance(max_steps, (int, float, dict))
    df = deepcopy(df)
    for i, row in df.iterrows():
        if isinstance(max_steps, dict):
            max_steps_for_env = max_steps.get(row['env_name'], float('inf'))
        else:
            max_steps_for_env = max_steps
        training_step = row['training_step']
        mask = np.where(training_step <= max_steps_for_env)
        for c in row.index:
            if (
                isinstance(row[c], np.ndarray)
                and row[c].ndim > 0
                and len(row[c]) == len(training_step)
            ):
                df.at[i, c] = row[c][mask]
    return df


def remove_incomplete(df, max_steps):
    """Returns a copy"""
    assert isinstance(max_steps, (int, float, dict))
    if isinstance(max_steps, dict):
        max_steps = defaultdict(lambda: float('inf'), max_steps)
    else:
        scalar_max_steps = int(max_steps)
        max_steps = defaultdict(lambda: scalar_max_steps)
    df = deepcopy(df)
    df['last_training_step'] = df['training_step'].apply(lambda x: x[-1])
    df['desired_max_steps'] = df['env_name'].map(max_steps)
    df = df.query('last_training_step >= desired_max_steps')
    df = df.drop(columns=['last_training_step', 'desired_max_steps'])
    return df
