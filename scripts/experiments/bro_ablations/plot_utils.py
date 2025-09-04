import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from qscaled.core.preprocessing import get_envs, get_utds
from qscaled.utils import plot_utils as qscaled_plot_utils
from rliable import plot_utils as rliable_plot_utils

np.random.seed(42)

from scripts.core.bootstrapping import _plot_optimal_hparam_fit_per_env_helper
from scripts.utils import abbreviate_number, expand_log_range


def get_model_sizes(df):
    if 'model_size' in df.columns:
        return sorted(df['model_size'].unique().tolist())
    else:
        return sorted(df['critic_params'].unique().tolist())


def learning_curves_per_utd(df, filtered_df=None, **kwargs):
    """
    By default (if `filtered_df is None`), assumes that df already filters out resets.
    Otherwise, df is unfiltered, and we plot the original returns with low opacity.
    """
    envs = get_envs(df)
    utds = get_utds(df)
    model_sizes = get_model_sizes(df)
    fig, axes = plt.subplots(len(envs), len(utds), figsize=(len(utds) * 3, 2.2 * len(envs)))
    axes = np.array(axes).reshape(len(envs), len(utds))

    colors = sns.color_palette('viridis', n_colors=len(model_sizes))
    lines, labels = [], []

    if filtered_df is None:
        filtered_df = df

    for i, env in enumerate(envs):
        for j, utd in enumerate(utds):
            ax = axes[i, j]
            for k, model_size in enumerate(model_sizes):
                label = f'model size {model_size}'
                filtered_subset = filtered_df.query(
                    f'env_name == "{env}" and utd == {utd} and model_size == {model_size}'
                )
                subset = df.query(
                    f'env_name == "{env}" and utd == {utd} and model_size == {model_size}'
                )
                assert len(subset) == 1
                unfiltered_data = subset.iloc[0]  # (Default) actually filtered
                data = filtered_subset.iloc[0]

                # ax.plot(
                #     unfiltered_data['training_step'],
                #     unfiltered_data['mean_return'],
                #     alpha=0.3,
                #     color=colors[k],
                # )
                ax.plot(
                    data['training_step_resetfilter'],
                    data['mean_return_resetfilter'],
                    alpha=0.3,
                    color=colors[k],
                )
                line = ax.plot(
                    data['training_step_resetfilter'],
                    data['return_isotonic'],
                    alpha=1,
                    color=colors[k],
                )

                # use the crossings column to plot crossings
                for t, threshold in enumerate(data['thresholds']):
                    crossing_x = data['crossings'][t]
                    crossing_y = threshold
                    ax.plot(
                        crossing_x,
                        crossing_y,
                        'o',
                        color=colors[k],
                    )

                # Plot crossing standard deviations as error bars
                for t, threshold in enumerate(data['thresholds']):
                    crossing_x = data['crossings'][t]
                    crossing_y = threshold
                    crossing_std = data['crossings_std'][t]
                    ax.errorbar(
                        crossing_x,
                        crossing_y,
                        xerr=crossing_std,
                        fmt='none',
                        color=colors[k],
                        capsize=3,
                    )

                if label not in labels:
                    lines.append(line[0])
                    labels.append(label)

            if i == 0:
                ax.set_title(f'UTD {utd}', fontsize=14)
            if j == 0:
                ax.set_ylabel(env, fontsize=kwargs.get('ylabel_fontsize', 14))
            ax.set_xlabel('env steps')
            ax.set_xlim(0, 1e6)
            ax.set_ylim(0, 1000)
            ax.grid(True, alpha=0.3)

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=(len(labels)),
        fontsize=12,
    )

    plt.tight_layout()


def learning_curves_per_model_size(df, filtered_df=None, **kwargs):
    envs = get_envs(df)
    utds = get_utds(df)
    model_sizes = get_model_sizes(df)
    fig, axes = plt.subplots(
        len(envs), len(model_sizes), figsize=(len(model_sizes) * 3, 2.2 * len(envs))
    )
    axes = np.array(axes).reshape(len(envs), len(model_sizes))

    colors = sns.color_palette('viridis', n_colors=len(utds))
    lines, labels = [], []

    if filtered_df is None:
        filtered_df = df

    for i, env in enumerate(envs):
        for j, model_size in enumerate(model_sizes):
            ax = axes[i, j]
            for k, utd in enumerate(utds):
                label = f'utd {utd}'
                filtered_subset = filtered_df.query(
                    f'env_name == "{env}" and utd == {utd} and model_size == {model_size}'
                )
                subset = df.query(
                    f'env_name == "{env}" and utd == {utd} and model_size == {model_size}'
                )
                assert len(subset) == 1
                unfiltered_data = subset.iloc[0]  # (Default) actually filtered
                data = filtered_subset.iloc[0]

                ax.plot(
                    unfiltered_data['training_step_resetfilter'],
                    unfiltered_data['mean_return_resetfilter'],
                    alpha=0.3,
                    color=colors[k],
                )
                line = ax.plot(
                    data['training_step_resetfilter'],
                    data['return_isotonic'],
                    alpha=1,
                    color=colors[k],
                )

                # use the crossings column to plot crossings
                for t, threshold in enumerate(data['thresholds']):
                    crossing_x = data['crossings'][t]
                    crossing_y = threshold
                    ax.plot(
                        crossing_x,
                        crossing_y,
                        'o',
                        color=colors[k],
                    )

                # Plot crossing standard deviations as error bars
                for t, threshold in enumerate(data['thresholds']):
                    crossing_x = data['crossings'][t]
                    crossing_y = threshold
                    crossing_std = data['crossings_std'][t]
                    ax.errorbar(
                        crossing_x,
                        crossing_y,
                        xerr=crossing_std,
                        fmt='none',
                        color=colors[k],
                        capsize=3,
                    )

                if label not in labels:
                    lines.append(line[0])
                    labels.append(label)

            if i == 0:
                ax.set_title(f'Model size {model_size}', fontsize=14)
            if j == 0:
                ax.set_ylabel(env, fontsize=kwargs.get('ylabel_fontsize', 14))
            ax.set_xlabel('env steps')
            ax.set_xlim(0, 1e6)
            ax.set_ylim(0, 1000)
            ax.grid(True, alpha=0.3)

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=(len(labels)),
        fontsize=12,
    )

    plt.tight_layout()


def compute_data_efficiency_per_env(df):
    """Compute the data efficiency dictionary for each environment."""
    data_efficiency_dict = {}

    envs = get_envs(df)
    utds = get_utds(df)

    for env in envs:
        env_df = df[df['env_name'] == env]
        utds = get_utds(env_df)
        model_sizes = get_model_sizes(env_df)
        outputs = []
        for utd in utds:
            for model_size in model_sizes:
                if 'model_size' in env_df.columns:
                    subset = env_df.query(f'utd=={utd} and model_size=={model_size}')
                else:
                    subset = env_df.query(f'utd=={utd} and critic_params=={model_size}')
                if len(subset) == 0:
                    print('no data for', (env, utd, model_size))
                    continue
                assert len(subset) == 1
                times = subset['crossings'].iloc[0][:]
                if np.isnan(times).any():
                    print(
                        f'env={env}, utd={utd}, model_size={model_size} has at least one nan'
                        + (
                            f'max return {subset["return_isotonic"].iloc[0].max()}'
                            if 'return_isotonic' in subset.columns
                            else ''
                        )
                    )
                outputs.append((utd, model_size, times))
        data_efficiency_dict[env] = outputs

    return data_efficiency_dict


def plot_data_efficiency_per_env(data_efficiency_dict, df, thresholds_per_env, threshold_idx=-1):
    """Plot the number of environment steps taken to achieve each performance threshold."""
    envs = get_envs(df)
    utds = get_utds(df)
    model_sizes = get_model_sizes(df)

    n_envs = len(envs)
    n_cols = 4
    n_rows = int(np.ceil(n_envs / n_cols))

    # Plot 1: Group by model size

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    fig.suptitle('Data Efficiency by Environment', fontsize=14)
    colors = sns.color_palette('viridis', n_colors=len(model_sizes))

    lines, labels = [], []

    for i, env in enumerate(envs):
        axes[i].set_xlabel('UTD')
        axes[i].set_ylabel('Env steps to Threshold')
        axes[i].set_title(f'{env} (threshold {round(thresholds_per_env[env][threshold_idx], 2)})')

        if env in data_efficiency_dict and len(data_efficiency_dict[env]) > 0:
            for j, model_size in enumerate(model_sizes):
                label = f'model size {model_size}'
                plot_utds = []
                times = []
                for utd, model_size_, times_ in data_efficiency_dict[env]:
                    if model_size_ == model_size and utd in utds:
                        plot_utds.append(utd)
                        times.append(times_[threshold_idx])
                plot_utds = np.array(plot_utds)
                times = np.array(times)

                if np.all(np.isnan(times)):
                    print(f'Skipping model size {model_size} for env {env} due to all NaN values.')
                    continue

                line = axes[i].plot(plot_utds, times, 'o-', label=label, color=colors[j])
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
                axes[i].grid(True, alpha=0.3)

                if label not in labels:
                    lines.append(line[0])
                    labels.append(label)

    for j in range(len(envs), len(axes)):
        axes[j].axis('off')

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=(len(labels)),
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    # Plot 2: Group by UTD

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    fig.suptitle('Data Efficiency by Environment', fontsize=14)
    colors = sns.color_palette('viridis', n_colors=len(utds))

    lines, labels = [], []

    for i, env in enumerate(envs):
        axes[i].set_xlabel('Model size')
        axes[i].set_ylabel('Env steps to Threshold')
        axes[i].set_title(f'{env} (threshold {round(thresholds_per_env[env][threshold_idx], 2)})')

        if env in data_efficiency_dict and len(data_efficiency_dict[env]) > 0:
            for j, utd in enumerate(utds):
                label = f'UTD {utd}'
                plot_model_sizes = []
                times = []
                for utd_, model_size, times_ in data_efficiency_dict[env]:
                    if utd_ == utd and model_size in model_sizes:
                        plot_model_sizes.append(model_size)
                        times.append(times_[threshold_idx])
                plot_model_sizes = np.array(plot_model_sizes)
                times = np.array(times)

                if np.all(np.isnan(times)):
                    print(f'Skipping UTD {utd} for env {env} due to all NaN values.')
                    continue

                line = axes[i].plot(plot_model_sizes, times, 'o-', label=label, color=colors[j])
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
                axes[i].grid(True, alpha=0.3)

                if label not in labels:
                    lines.append(line[0])
                    labels.append(label)

    for j in range(len(envs), len(axes)):
        axes[j].axis('off')

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=(len(labels)),
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def compute_normalized_times(data_efficiency_dict, df, n_thresholds):
    """
    Compute normalized times and scaling factors for each environment.
    Implements Appendix D of the paper.
    """
    envs = get_envs(df)
    utds = get_utds(df)
    model_sizes = get_model_sizes(df)

    median_times = []
    for env in envs:
        env_times = []
        for utd, model_size, times in data_efficiency_dict[env]:
            if utd not in utds or model_size not in model_sizes:
                continue
            env_times.extend(times)
        median_times.append(np.nanmedian(env_times))
    median_times = np.array(median_times)
    median_median = np.nanmedian(median_times)
    scaling = 1 / median_times
    normalized_times_all = np.full((len(envs), len(utds), len(model_sizes), n_thresholds), np.nan)
    for i, env in enumerate(envs):
        if len(data_efficiency_dict[env]) > 0:
            for utd, model_size, times in data_efficiency_dict[env]:
                if env not in envs or utd not in utds or model_size not in model_sizes:
                    continue
                utd_idx = utds.index(utd)
                model_size_idx = model_sizes.index(model_size)
                normalized_times_all[i, utd_idx, model_size_idx] = np.array(times) * scaling[i]
    mean_normalized_times = np.nanmean(normalized_times_all, axis=0)
    return np.array(normalized_times_all), mean_normalized_times, median_median


def plot_data_efficiency_averaged(mean_normalized_times, median_median, df, threshold_idx=-1):
    def set_layout():
        try:
            plt.tight_layout()
        except ValueError as e:
            if 'Data has no positive values' in str(e):
                warnings.warn(
                    'Matplotlib error coming. In the per-env data efficiency plot, check that each performance '
                    'threshold is achieved for every UTD. If not, decrease your thresholds in the config, and '
                    'call `bootstrap_crossings` with `use_cached=False`.',
                    UserWarning,
                )
            raise e

    utds = get_utds(df)
    model_sizes = get_model_sizes(df)

    # Group by model size

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = sns.color_palette('viridis', n_colors=len(model_sizes))

    for j, model_size in enumerate(model_sizes):
        label = f'model size {model_size}'
        times = mean_normalized_times[:, j, threshold_idx] * median_median
        ax.plot(utds, times, 'o-', label=label, color=colors[j])

    ax.set_title(f'Averaged data efficiency, threshold_idx {threshold_idx}')
    ax.set_xscale('log')
    ax.set_xlabel('UTD')
    ax.set_ylabel('Env steps to Threshold')
    ax.grid(True, alpha=0.3)
    fig.legend(bbox_to_anchor=(0.5, 0), loc='upper center', ncol=2, fontsize=12)
    set_layout()
    plt.show()

    # Group by UTD

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = sns.color_palette('viridis', n_colors=len(utds))

    for i, utd in enumerate(utds):
        label = f'UTD={utd}'
        times = mean_normalized_times[i, :, threshold_idx] * median_median
        ax.plot(model_sizes, times, 'o-', label=label, color=colors[i])

    ax.set_xscale('log')
    ax.set_title(f'Averaged data efficiency, threshold_idx {threshold_idx}')
    ax.set_xlabel('Model size')
    ax.set_ylabel('Env steps to Threshold')
    ax.grid(True, alpha=0.3)
    fig.legend(bbox_to_anchor=(0.5, 0), loc='upper center', ncol=3, fontsize=12)
    set_layout()
    plt.show()


def plot_average_return_per_env(df):
    envs = get_envs(df)
    utds = get_utds(df)
    model_sizes = get_model_sizes(df)

    n_envs = len(envs)
    n_cols = 4
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    fig.suptitle('Average return by environment', fontsize=14)
    colors = sns.color_palette('viridis', n_colors=len(model_sizes))
    lines, labels = [], []

    for i, (env, ax) in enumerate(zip(envs, axes)):
        for j, model_size in enumerate(model_sizes):
            label = f'model size {model_size}'
            subset = df.query(f'env_name == "{env}" and model_size == {model_size}')
            subset.sort_values('utd', inplace=True)
            line = ax.plot(
                subset['utd'],
                subset['mean_return_over_training'],
                'o-',
                label=label,
                color=colors[j],
            )
            ax.fill_between(
                subset['utd'],
                subset['mean_return_over_training'] - subset['std_return_over_training'],
                subset['mean_return_over_training'] + subset['std_return_over_training'],
                color=colors[j],
                alpha=0.2,
            )
            if label not in labels:
                lines.append(line[0])
                labels.append(label)

        ax.set_title(env)
        ax.set_xscale('log')
        ax.set_xlabel('UTD')
        ax.set_ylabel('Average Return')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1000)

    for j in range(len(envs), len(axes)):
        axes[j].axis('off')

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=(len(labels)),
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    fig.suptitle('Average return by environment', fontsize=14)
    colors = sns.color_palette('viridis', n_colors=len(utds))
    lines, labels = [], []

    for i, (env, ax) in enumerate(zip(envs, axes)):
        for j, utd in enumerate(utds):
            label = f'UTD={utd}'
            subset = df.query(f'env_name == "{env}" and utd == {utd}')
            subset.sort_values('model_size', inplace=True)
            line = ax.plot(
                subset['model_size'],
                subset['mean_return_over_training'],
                'o-',
                label=label,
                color=colors[j],
            )
            ax.fill_between(
                subset['model_size'],
                subset['mean_return_over_training'] - subset['std_return_over_training'],
                subset['mean_return_over_training'] + subset['std_return_over_training'],
                color=colors[j],
                alpha=0.2,
            )
            if label not in labels:
                lines.append(line[0])
                labels.append(label)

        ax.set_title(env)
        ax.set_xscale('log')
        ax.set_xlabel('Model size')
        ax.set_ylabel('Average Return')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1000)

    for j in range(len(envs), len(axes)):
        axes[j].axis('off')

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=(len(labels)),
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_average_return_averaged(df):
    envs = get_envs(df)
    utds = get_utds(df)
    model_sizes = get_model_sizes(df)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle('Average return, averaged over environments')
    colors = sns.color_palette('viridis', n_colors=len(model_sizes))
    lines, labels = [], []

    for j, model_size in enumerate(model_sizes):
        label = f'model size {model_size}'
        subset = df.query(f'model_size == {model_size}')
        mean_returns_by_utd = subset.groupby('utd')['mean_return_over_training'].mean()
        std_returns_by_utd = subset.groupby('utd')['std_return_over_training'].mean() / np.sqrt(
            len(envs)
        )
        line = ax.plot(
            utds,
            mean_returns_by_utd,
            'o-',
            label=label,
            color=colors[j],
        )
        ax.fill_between(
            utds,
            mean_returns_by_utd - std_returns_by_utd,
            mean_returns_by_utd + std_returns_by_utd,
            color=colors[j],
            alpha=0.2,
        )
        if label not in labels:
            lines.append(line[0])
            labels.append(label)

    plt.xscale('log')
    plt.xlabel('UTD')
    plt.ylabel('Average Return')

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=2,
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle('Average return, averaged over environments')
    colors = sns.color_palette('viridis', n_colors=len(model_sizes))
    lines, labels = [], []

    for j, utd in enumerate(utds):
        label = f'UTD={utd}'
        subset = df.query(f'utd == {utd}')
        mean_returns_by_utd = subset.groupby('model_size')['mean_return_over_training'].mean()
        std_returns_by_utd = subset.groupby('model_size')[
            'std_return_over_training'
        ].mean() / np.sqrt(len(envs))
        line = ax.plot(
            model_sizes,
            mean_returns_by_utd,
            'o-',
            label=label,
            color=colors[j],
        )
        ax.fill_between(
            model_sizes,
            mean_returns_by_utd - std_returns_by_utd,
            mean_returns_by_utd + std_returns_by_utd,
            color=colors[j],
            alpha=0.2,
        )
        if label not in labels:
            lines.append(line[0])
            labels.append(label)

    plt.xscale('log')
    plt.xlabel('Model size')
    plt.ylabel('Average Return')

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='upper center',
        ncol=3,
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_averaged_threshold_fits_per_model_size(threshold_dfs, predict_fn_info_pairs):
    for threshold_idx, (threshold_df, predict_fn_info_pairs_) in enumerate(
        zip(threshold_dfs, predict_fn_info_pairs)
    ):
        _plot_optimal_hparam_fit_per_env_helper(
            threshold_df,
            group_col='critic_params',
            group_label_col='critic_params',
            xcol='utd',
            ycol='time',
            ycol_std='time_std',
            title=f'threshold {threshold_idx}',
            predict_fn_info_pairs=predict_fn_info_pairs_,
            title_fn=lambda x: f'$N$={abbreviate_number(x)}',
            xlabel_fn=lambda x: int(x),
            data_label='Time to threshold',
        )


def plot_averaged_threshold_fits_per_utd(threshold_dfs, predict_fn_info_pairs):
    for threshold_idx, (threshold_df, predict_fn_info_pairs_) in enumerate(
        zip(threshold_dfs, predict_fn_info_pairs)
    ):
        _plot_optimal_hparam_fit_per_env_helper(
            threshold_df,
            group_col='utd',
            group_label_col='utd',
            xcol='critic_params',
            ycol='time',
            ycol_std='time_std',
            title=f'threshold {threshold_idx}',
            predict_fn_info_pairs=predict_fn_info_pairs_,
            title_fn=lambda x: f'$\sigma$={int(x)}',
            xlabel_fn=abbreviate_number,
            xlabel_fontsize=8,
            xlabel_rotation=25,
            data_label='Time to threshold',
        )


def _plot_optimal_hparam_fit_per_env_multiple_thresh_helper(
    threshold_dfs,
    group_col,
    group_label_col,
    xcol,
    ycol,
    ycol_std,
    title,
    predict_fn_info_pairs,
    title_fn,
    xlabel_fn,
    **kw,
):
    n_thresholds = len(threshold_dfs)
    assert len(predict_fn_info_pairs) == n_thresholds
    sample_df = threshold_dfs[0]
    envs = get_envs(sample_df)
    all_xs = sorted(sample_df[xcol].unique())
    all_group_vals = sorted(sample_df[group_col].unique())
    x_smooth = np.logspace(np.log10(min(all_xs)), np.log10(max(all_xs)), 100)

    def set_yscale(ax):
        if kw.get('yscale') == 'log2':
            ax.set_yscale('log', base=2)
        elif kw.get('yscale') == 'log':
            ax.set_yscale('log')

    fig, axes = plt.subplots(
        len(envs),
        len(all_group_vals),
        figsize=(4 * len(all_group_vals), 3.5 * len(envs)),
        # sharey=True,
    )
    axes = np.array(axes).reshape(len(envs), len(all_group_vals))
    handles, labels = [], []
    # xmin, xmax = float('inf'), float('-inf')
    # ymin, ymax = float('inf'), float('-inf')

    colors = sns.color_palette('viridis', n_colors=n_thresholds)

    xticks_kw = {}
    if 'xlabel_rotation' in kw:
        xticks_kw['rotation'] = kw['xlabel_rotation']
    if 'xlabel_fontsize' in kw:
        xticks_kw['fontsize'] = kw['xlabel_fontsize']
    hard_ymin = kw.get('hard_ymin', float('-inf'))
    hard_ymax = kw.get('hard_ymax', float('inf'))

    lines = []
    labels = []

    for i, env in enumerate(envs):
        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')

        for threshold_idx, (threshold_df, predict_fn_info_pair, color) in enumerate(
            zip(threshold_dfs, predict_fn_info_pairs, colors), 1
        ):
            label = f'Threshold {threshold_idx}'
            env_df = threshold_df.query(f'env_name=="{env}"')

            for j, group_val in enumerate(all_group_vals):
                ax = axes[i, j]
                dta = env_df.query(f'{group_col}=={group_val}').sort_values(xcol)
                if len(dta) == 0:
                    continue
                assert dta[group_label_col].nunique() == 1
                group_label_val = dta[group_label_col].iloc[0]
                prediction_input_df = pd.DataFrame(
                    {'env_name': env, xcol: x_smooth, group_label_col: group_label_val}
                )

                ax.errorbar(
                    dta[xcol],
                    dta[ycol],
                    yerr=dta[ycol_std] if ycol_std else None,
                    fmt='o',
                    color=color,
                    capsize=3,
                )
                predict_fn, info = predict_fn_info_pair
                try:
                    fit_predictions = predict_fn(prediction_input_df)
                except:
                    continue
                # prediction_input_df[f'predictions_{label}'] = fit_predictions

                # use color from `colors`, not `predict_fn_info_pairs_`
                (line,) = ax.plot(x_smooth, fit_predictions, label=label, color=color)
                ymin = min(ymin, max(min(fit_predictions), hard_ymin))
                ymax = max(ymax, min(max(fit_predictions), hard_ymin))

                if 'asymptote' in info:
                    asymptote = info['asymptote'][(env, group_label_val)]
                    if asymptote > hard_ymin and asymptote < hard_ymax:
                        ax.axhline(y=asymptote, color=color, linestyle='--', zorder=-1)
                        ymin = min(ymin, asymptote)
                        ymax = max(ymax, asymptote)

                if env != '':
                    ax.set_title(f'{env}, {title_fn(group_label_val)}')
                else:
                    ax.set_title(title_fn(group_label_val))
                ax.set_xlabel(xcol)
                ax.set_xscale('log', base=2)
                set_yscale(ax)
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_xticks(dta[xcol])
                ax.set_xticklabels([xlabel_fn(x) for x in dta[xcol]], **xticks_kw)
                if kw.get('yscale') == 'log2':
                    ax.set_yscale('log', base=2)
                else:
                    ax.set_yscale('log')
                ax.grid(alpha=0.3)
                xmin, xmax = min(xmin, dta[xcol].min()), max(xmax, dta[xcol].max())
                ymin, ymax = min(ymin, dta[ycol].min()), max(ymax, dta[ycol].max())

            if label not in labels:
                labels.append(label)
                lines.append(line)

        for ax in axes[i]:
            ax.set_xlim(*expand_log_range(xmin, xmax))
            ax.set_ylim(*expand_log_range(ymin, ymax))

    # for ax in axes.flatten():
    #     ax.set_xlim(*expand_log_range(xmin, xmax))
    #     ax.set_ylim(*expand_log_range(ymin, ymax))

    sorted_indices = [
        i for i, _ in sorted(enumerate(labels), key=lambda x: int(x[1].split(' ')[-1]))
    ]
    lines = [lines[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, 0),
        ncol=len(all_group_vals),
        loc='upper center',
        fontsize=14,
    )
    fig.suptitle(f'{title}, grouped by {group_label_col}', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_optimal_hparam_fit_per_env_n_multiple_thresh(
    threshold_dfs, ycol, ycol_std, title, predict_fn_info_pairs, **kw
):
    _plot_optimal_hparam_fit_per_env_multiple_thresh_helper(
        threshold_dfs,
        group_col=kw.pop('group_col', 'critic_params'),
        group_label_col='critic_params',
        xcol='utd',
        ycol=ycol,
        ycol_std=ycol_std,
        title=title,
        predict_fn_info_pairs=predict_fn_info_pairs,
        title_fn=lambda x: f'$N$={abbreviate_number(x)}',
        xlabel_fn=lambda x: int(x),
        **kw,
    )


def plot_optimal_hparam_fit_per_env_utd_multiple_thresh(
    threshold_dfs, ycol, ycol_std, title, predict_fn_info_pairs, **kw
):
    _plot_optimal_hparam_fit_per_env_multiple_thresh_helper(
        threshold_dfs,
        group_col='utd',
        group_label_col='utd',
        xcol='critic_params',
        ycol=ycol,
        ycol_std=ycol_std,
        title=title,
        predict_fn_info_pairs=predict_fn_info_pairs,
        title_fn=lambda x: f'$\sigma$={int(x)}',
        xlabel_fn=abbreviate_number,
        xlabel_fontsize=8,
        xlabel_rotation=25,
        **kw,
    )


def _plot_optimal_hparam_fit_per_env_multiple_thresh_helper_pretty(
    threshold_dfs,
    group_col,
    group_label_col,
    xcol,
    ycol,
    ycol_std,
    title,
    thresholds_per_env,
    predict_fn_info_pairs,
    title_fn,
    xlabel_fn,
    **kw,
):
    n_thresholds = len(threshold_dfs)
    assert len(predict_fn_info_pairs) == n_thresholds
    sample_df = threshold_dfs[0]
    envs = get_envs(sample_df)
    all_xs = sorted(sample_df[xcol].unique())
    all_group_vals = sorted(sample_df[group_col].unique())
    x_smooth = np.logspace(np.log10(min(all_xs)), np.log10(max(all_xs)), 100)

    def set_yscale(ax):
        if kw.get('yscale') == 'log2':
            ax.set_yscale('log', base=2)
        elif kw.get('yscale') == 'log':
            ax.set_yscale('log')

    fig, axes = plt.subplots(
        len(envs),
        len(all_group_vals),
        figsize=(5 * len(all_group_vals), 4 * len(envs)),
        # layout='constrained',
        # sharey=True,
    )
    axes = np.array(axes).reshape(len(envs), len(all_group_vals))
    handles, labels = [], []
    # xmin, xmax = float('inf'), float('-inf')
    # ymin, ymax = float('inf'), float('-inf')

    # colors = sns.color_palette('viridis', n_colors=n_thresholds)
    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [qscaled_plot_utils.COLORS[0], qscaled_plot_utils.COLORS[1]]
    )
    colors = [cmap(i) for i in np.linspace(0, 1, n_thresholds)]

    xticks_kw = {}
    if 'xlabel_rotation' in kw:
        xticks_kw['rotation'] = kw['xlabel_rotation']
    if 'xlabel_fontsize' in kw:
        xticks_kw['fontsize'] = kw['xlabel_fontsize']
    hard_ymin = kw.get('hard_ymin', float('-inf'))
    hard_ymax = kw.get('hard_ymax', float('inf'))

    lines = []
    labels = []

    for i, env in enumerate(envs):
        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')

        for threshold_idx, (threshold_df, predict_fn_info_pair, color) in enumerate(
            zip(threshold_dfs, predict_fn_info_pairs, colors), 1
        ):
            # label = f'Threshold {threshold_idx}'
            env_df = threshold_df.query(f'env_name=="{env}"')

            for j, group_val in enumerate(all_group_vals):
                ax = axes[i, j]
                dta = env_df.query(f'{group_col}=={group_val}').sort_values(xcol)
                if len(dta) == 0:
                    continue
                assert dta[group_label_col].nunique() == 1
                group_label_val = dta[group_label_col].iloc[0]
                prediction_input_df = pd.DataFrame(
                    {'env_name': env, xcol: x_smooth, group_label_col: group_label_val}
                )

                ax.errorbar(
                    dta[xcol],
                    dta[ycol],
                    # yerr=dta[ycol_std] if ycol_std else None,
                    fmt='o',
                    color=color,
                    capsize=3,
                )
                predict_fn, info = predict_fn_info_pair
                try:
                    fit_predictions = predict_fn(prediction_input_df)
                except:
                    continue
                # prediction_input_df[f'predictions_{label}'] = fit_predictions

                # use color from `colors`, not `predict_fn_info_pairs_`
                (line,) = ax.plot(x_smooth, fit_predictions, color=color)
                ymin = min(ymin, max(min(fit_predictions), hard_ymin))
                ymax = max(ymax, min(max(fit_predictions), hard_ymin))

                # if 'asymptote' in info:
                if False:
                    asymptote = info['asymptote'][(env, group_label_val)]
                    if asymptote > hard_ymin and asymptote < hard_ymax:
                        ax.axhline(y=asymptote, color=color, linestyle='--', zorder=-1)
                        ymin = min(ymin, asymptote)
                        ymax = max(ymax, asymptote)

                if env != '':
                    ax.set_title(f'{env}, {title_fn(group_label_val)}', fontsize='xx-large')
                else:
                    ax.set_title(title_fn(group_label_val), fontsize='xx-large')
                ax.set_xlabel(xcol)
                ax.set_xscale('log', base=2)
                set_yscale(ax)
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_xticks(dta[xcol])
                ax.set_xticklabels([xlabel_fn(x) for x in dta[xcol]], **xticks_kw)
                if kw.get('yscale') == 'log2':
                    ax.set_yscale('log', base=2)
                else:
                    ax.set_yscale('log')
                ax.grid(alpha=0.3)
                xmin, xmax = min(xmin, dta[xcol].min()), max(xmax, dta[xcol].max())
                ymin, ymax = min(ymin, dta[ycol].min()), max(ymax, dta[ycol].max())

            # if label not in labels:
            #     labels.append(label)
            #     lines.append(line)

        for ax in axes[i]:
            ax.set_xlim(*expand_log_range(xmin, xmax))
            ax.set_ylim(*expand_log_range(ymin, ymax))

    for i, env in enumerate(envs):
        for j in range(len(all_group_vals)):
            ax = axes[i, j]
            rliable_plot_utils._annotate_and_decorate_axis(
                ax,
                xlabel=(
                    r'$\sigma$: UTD' if group_label_col == 'critic_params' else r'$N$: Model size'
                )
                if i == len(envs) - 1
                else '',
                ylabel=r'$\mathcal{D}$: Data' if j == 0 else '',
                labelsize='xx-large',
                ticklabelsize='xx-large',
                grid_alpha=0.2,
                legend=False,
            )
            if 'yticks' in kw:
                qscaled_plot_utils.ax_set_y_bounds_and_scale(
                    ax, yticks=kw['yticks'][env], yscale='1e5'
                )

    cbar_left = 0.91
    cbar_width = 0.01
    top_margin = 0.9
    bottom_margin = 0.1
    vertical_gap = 0.02
    row_height = (top_margin - bottom_margin - (len(envs) - 1) * vertical_gap) / len(envs)

    for i, env in enumerate(envs):
        bottom = top_margin - (i + 1) * row_height - i * vertical_gap
        cax = fig.add_axes([cbar_left, bottom, cbar_width, row_height])
        norm = plt.Normalize(vmin=thresholds_per_env[env][0], vmax=thresholds_per_env[env][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('$J$: Return', size='xx-large')
        cbar.ax.tick_params(labelsize='xx-large')

    # cax = fig.add_axes([0.91, 0.15, 0.01, 0.7])  # Adjust as needed
    # cbar = plt.colorbar(sm, cax=cax)
    # cbar.set_label(r'$J$: Return', size='xx-large')
    # cbar.ax.tick_params(labelsize='xx-large')

    # for ax in axes.flatten():
    #     ax.set_xlim(*expand_log_range(xmin, xmax))
    #     ax.set_ylim(*expand_log_range(ymin, ymax))

    # sorted_indices = [
    #     i for i, _ in sorted(enumerate(labels), key=lambda x: int(x[1].split(' ')[-1]))
    # ]
    # lines = [lines[i] for i in sorted_indices]
    # labels = [labels[i] for i in sorted_indices]

    # fig.legend(
    #     lines,
    #     labels,
    #     bbox_to_anchor=(0.5, 0),
    #     ncol=len(all_group_vals),
    #     loc='upper center',
    #     fontsize=14,
    # )
    # fig.suptitle(f'{title}, grouped by {group_label_col}', y=1.02, fontsize=16)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    # plt.tight_layout()

    if 'save_path' in kw:
        os.makedirs(os.path.dirname(kw['save_path']), exist_ok=True)
        plt.savefig(kw['save_path'], bbox_inches='tight')

    plt.show()


def plot_optimal_hparam_fit_per_env_n_multiple_thresh_pretty(
    threshold_dfs, ycol, ycol_std, title, thresholds_per_env, predict_fn_info_pairs, **kw
):
    _plot_optimal_hparam_fit_per_env_multiple_thresh_helper_pretty(
        threshold_dfs,
        group_col=kw.pop('group_col', 'critic_params'),
        group_label_col='critic_params',
        xcol='utd',
        ycol=ycol,
        ycol_std=ycol_std,
        title=title,
        thresholds_per_env=thresholds_per_env,
        predict_fn_info_pairs=predict_fn_info_pairs,
        title_fn=lambda x: f'$N$={abbreviate_number(x)}',
        xlabel_fn=lambda x: int(x),
        **kw,
    )


def plot_optimal_hparam_fit_per_env_utd_multiple_thresh_pretty(
    threshold_dfs, ycol, ycol_std, title, thresholds_per_env, predict_fn_info_pairs, **kw
):
    _plot_optimal_hparam_fit_per_env_multiple_thresh_helper_pretty(
        threshold_dfs,
        group_col='utd',
        group_label_col='utd',
        xcol='critic_params',
        ycol=ycol,
        ycol_std=ycol_std,
        title=title,
        thresholds_per_env=thresholds_per_env,
        predict_fn_info_pairs=predict_fn_info_pairs,
        title_fn=lambda x: f'$\sigma$={int(x)}',
        xlabel_fn=abbreviate_number,
        xlabel_fontsize=8,
        xlabel_rotation=25,
        **kw,
    )
