import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import statsmodels.api as sm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import colorsys
from matplotlib import colormaps
from functools import partial
from scripts.utils import abbreviate_number, clean_sci, expand_log_range
from typing import Dict
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from rliable import plot_utils as rliable_plot_utils
from qscaled.utils import plot_utils as qscaled_plot_utils

from scripts.core.fitting import (
    sum_of_powerlaw_fn,
    softplus,
    _generic_fit_scipy,
    _log_rescale,
    _log_rescale_inverse,
    r_squared,
)

# Custom color palette
COLORS = [
    '#BBCC33',
    '#77AADD',
    '#44BB99',
    '#EEDD88',
    '#EE8866',
    '#FFAABB',
    '#99DDFF',
    '#AAAA00',
    '#DDDDDD',
]


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return mcolors.to_hex(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))


def centered_gradient(hex_color, n=5, reverse=False, light_extent=0.4, dark_extent=1):
    """Returns light-to-dark gradient as list of hex codes; reverse=True returns dark-to-light"""
    light_color = lighten_color(hex_color, 1 - light_extent)
    dark_color = lighten_color(hex_color, 1 + dark_extent)
    cmap = LinearSegmentedColormap.from_list('gradient', [light_color, dark_color])
    gradient_colors = [cmap(i / (n - 1)) for i in range(n)]
    gradient_colors = [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]
    if reverse:
        return gradient_colors[::-1]
    else:
        return gradient_colors


def plot_learning_curves_with_thresholds(
    df,
    var_name,
    thresholds,
    row_attr_name='critic_width',
    col_attr_name='utd',
    return_isotonic_key='return_isotonic',
    mean_return_key='mean_return',
    std_return_key=None,
    hline=True,
    hline_y=1000,
):
    envs = sorted(df['env_name'].unique())
    if row_attr_name is not None:
        row_attr_values = sorted(df[row_attr_name].unique())
    else:
        row_attr_values = [None]
    if col_attr_name is not None:
        col_attr_values = sorted(df[col_attr_name].unique())
    else:
        col_attr_values = [None]
    var_values = sorted(df[var_name].unique())
    colors = sns.color_palette('viridis', len(var_values))

    all_figs = {}

    crossings_warning = False

    for env in envs:
        if isinstance(thresholds, list):
            env_thresholds = thresholds
        elif isinstance(thresholds, (dict, pd.Series)):
            env_thresholds = thresholds[env]
        else:
            raise ValueError

        fig, axes = plt.subplots(
            len(row_attr_values),
            len(col_attr_values),
            figsize=(len(col_attr_values) * 3, len(row_attr_values) * 2.5),
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).reshape(len(row_attr_values), len(col_attr_values))

        lines, labels = [], []

        for i, row_attr_value in enumerate(row_attr_values):
            for j, col_attr_value in enumerate(col_attr_values):
                ax = axes[i, j]
                title = []
                if row_attr_name is not None:
                    title.append(f'{row_attr_name}: {row_attr_value}')
                if col_attr_name is not None:
                    title.append(f'{col_attr_name}: {col_attr_value}')
                ax.set_title(', '.join(title))
                if hline:
                    ax.axhline(y=hline_y, color='gray', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)

                subset = df.query(f'env_name=="{env}"')
                if row_attr_name is not None:
                    subset = subset.query(f'{row_attr_name}=={row_attr_value}')
                if col_attr_name is not None:
                    subset = subset.query(f'{col_attr_name}=={col_attr_value}')

                for _, row in subset.iterrows():
                    color = colors[var_values.index(row[var_name])]
                    label = f'{var_name}={row[var_name]}'

                    ax.plot(
                        row['training_step'],
                        row[mean_return_key],
                        color=color,
                        alpha=0.3,
                    )
                    if std_return_key is not None:
                        ax.fill_between(
                            row['training_step'],
                            row[mean_return_key] - row[std_return_key],
                            row[mean_return_key] + row[std_return_key],
                            alpha=0.2,
                            color=color,
                            linewidth=0,
                        )

                    if 'training_step_resetfilter' in row:
                        training_step = row['training_step_resetfilter']
                    else:
                        training_step = row['training_step']

                    line = ax.plot(
                        training_step,
                        row[return_isotonic_key],
                        color=color,
                        alpha=1,
                        label=label,
                    )

                    if label not in labels:
                        labels.append(label)
                        lines.append(line[0])

                    if 'crossings' not in row:
                        if not crossings_warning:
                            print('Warning: crossings not found')
                            crossings_warning = True
                        continue

                    # use the crossings column to plot crossings
                    for k, threshold in enumerate(env_thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        ax.plot(crossing_x, crossing_y, 'o', color=color)

                    # Plot crossing standard deviations as error bars
                    for k, threshold in enumerate(env_thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        crossing_std = row['crossings_std'][k]
                        ax.errorbar(
                            crossing_x,
                            crossing_y,
                            xerr=crossing_std,
                            capsize=3,
                            color=color,
                        )

        idx = [i for i, _ in sorted(enumerate(labels), key=lambda x: float(x[1].split('=')[1]))]
        labels = [labels[i] for i in idx]
        lines = [lines[i] for i in idx]

        fig.legend(
            lines,
            labels,
            loc='upper center',
            ncol=2,
            fontsize=12,
            bbox_to_anchor=(0.5, 0),
        )
        title = f'{env}: {var_name}'
        if len(env_thresholds) > 0:
            title += f', max threshold {round(env_thresholds[-1], 2)}'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        all_figs[env] = fig

    return all_figs


def plot_optimal_hparams_heatmap(best_hparams, key: str, group_how=('critic_params', 'utd')):
    envs = sorted(best_hparams['env_name'].unique())
    n_envs = len(envs)
    n_cols = min(4, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).reshape(-1)
    if 'lr' in key:
        fmt = '.2e'
    elif 'time' in key:
        fmt = '.2e'
    else:
        fmt = '.0f'

    batch_size = 256  # TODO: make this dynamic

    for i, (env, ax) in enumerate(zip(envs, axes)):
        env_data = best_hparams[best_hparams['env_name'] == env]
        env_data['compute'] = 10 * batch_size * env_data['utd'] * env_data['critic_params']
        pivot = env_data.pivot(index=group_how[1], columns=[group_how[0]], values=key).sort_index(
            ascending=False
        )
        if group_how[0] == 'critic_params':
            pivot.columns = pivot.columns.map(abbreviate_number)
        if 'e' in fmt:
            formatted_annot = pivot.map(clean_sci)
            sns.heatmap(
                pivot,
                annot=formatted_annot,
                fmt='',
                cmap='viridis',
                ax=ax,
                cbar=True,
                norm=LogNorm(),
            )
        else:
            sns.heatmap(
                pivot,
                annot=True,
                fmt=fmt,
                cmap='viridis',
                ax=ax,
                cbar=True,
                norm=LogNorm(),
            )
        ax.set_title(env)
        ax.set_aspect('equal')

    for i in range(len(envs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(key, y=1.05)
    plt.show()


def naive_cost(env, utd, critic_params, predict_fn):
    batch_size = 256
    return 10 * batch_size * utd * critic_params


def _data_compute_fn(env, utd, critic_params, predict_fn):
    assert isinstance(env, str)
    batch_size = 256
    utds_flat = utd.flatten()
    critic_params_flat = critic_params.flatten()
    input_df = pd.DataFrame({'utd': utds_flat, 'critic_params': critic_params_flat})
    input_df['env_name'] = env
    data_efficiency = np.array(predict_fn(input_df)).reshape(utd.shape)
    return data_efficiency, 10 * batch_size * utd * critic_params * data_efficiency


def d_cost(env, utd, critic_params, predict_fn):
    return _data_compute_fn(env, utd, critic_params, predict_fn)[0]


def c_cost(env, utd, critic_params, predict_fn):
    return _data_compute_fn(env, utd, critic_params, predict_fn)[1]


def c_delta_d_cost(delta, env, utd, critic_params, predict_fn):
    res = _data_compute_fn(env, utd, critic_params, predict_fn)
    return res[1] + delta * res[0]


def _budget_fn(
    utd_and_critic_params,
    batch_size,
    delta,
    a,
    alpha,
    b,
    beta,
    c,
    utd_m,
    utd_s,
    critic_params_m,
    critic_params_s,
):
    u, v = utd_and_critic_params
    utd = _log_rescale_inverse(softplus(u), utd_m, utd_s)
    critic_params = _log_rescale_inverse(softplus(v), critic_params_m, critic_params_s)
    data = sum_of_powerlaw_fn(utd, critic_params, a, alpha, b, beta, c)
    compute = 10 * batch_size * utd * critic_params * data
    return compute + delta * data


def compute_optimal_budget(
    df, params_per_thresh, data_efficiency_key_prefix: str, delta: float | Dict, parallel=False
):
    batch_size = 256  # TODO: make this dynamic
    envs = sorted(df['env_name'].unique())
    n_thresholds = len(params_per_thresh)
    data_efficiency_keys = [f'{data_efficiency_key_prefix}{i}' for i in range(n_thresholds)]

    df = deepcopy(df)
    df.dropna(subset=data_efficiency_keys, inplace=True)
    for data_efficiency_key in data_efficiency_keys:
        df[f'compute_{data_efficiency_key}'] = (
            10 * batch_size * df['utd'] * df['critic_params'] * df[data_efficiency_key]
        )
        if isinstance(delta, dict):
            df[f'budget_{data_efficiency_key}'] = (
                df[f'compute_{data_efficiency_key}']
                + df['env_name'].map(delta) * df[data_efficiency_key]
            )
        else:
            df[f'budget_{data_efficiency_key}'] = (
                df[f'compute_{data_efficiency_key}'] + delta * df[data_efficiency_key]
            )

    optima = []

    for env in envs:
        env_delta = delta[env] if isinstance(delta, dict) else delta
        env_data = df[df['env_name'] == env]

        def process_threshold(threshold_idx, params):
            a, alpha, b, beta, c = params[env][:5]
            _, utd_m, utd_s = _log_rescale(np.array(env_data['utd'].values))
            _, critic_params_m, critic_params_s = _log_rescale(
                np.array(env_data['critic_params'].values)
            )
            init_grid = [slice(0.0, 0.0, 1.0), slice(0.0, 0.0, 1.0)]
            args = (
                batch_size,
                env_delta,
                a,
                alpha,
                b,
                beta,
                c,
                utd_m,
                utd_s,
                critic_params_m,
                critic_params_s,
            )
            u, v = _generic_fit_scipy(_budget_fn, args, init_grid, precise=True)
            opt_utd = _log_rescale_inverse(softplus(u), utd_m, utd_s)
            opt_critic_params = _log_rescale_inverse(softplus(v), critic_params_m, critic_params_s)
            data_efficiency = sum_of_powerlaw_fn(opt_utd, opt_critic_params, a, alpha, b, beta, c)
            compute = 10 * batch_size * opt_utd * opt_critic_params * data_efficiency
            budget = compute + env_delta * data_efficiency
            return (
                env,
                env_delta,
                threshold_idx,
                budget,
                opt_utd,
                opt_critic_params,
                data_efficiency,
                compute,
            )

        if parallel:
            raise NotImplementedError
            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     futures = [
            #         executor.submit(process_threshold, threshold_idx, predict_fn)
            #         for threshold_idx, predict_fn in enumerate(predict_fns)
            #     ]
            #     for future in tqdm(as_completed(futures), total=n_thresholds):
            #         optima.append(future.result())
        else:
            for threshold_idx, (data_efficiency_key, params) in tqdm(
                enumerate(zip(data_efficiency_keys, params_per_thresh)), total=n_thresholds
            ):
                optima.append(process_threshold(threshold_idx, params))

    return pd.DataFrame(
        optima,
        columns=[
            'env_name',
            'delta',
            'threshold_idx',
            'opt_budget',
            'opt_utd',
            'opt_critic_params',
            'data_efficiency',
            'compute',
        ],
    )


def plot_optimal_hparams_scatter(df, predict_fn, data_efficiency_key: str, mode: str, **kw):
    assert mode.split('_')[0] in ['data', 'compute', 'budget']
    assert mode.split('_')[1] in ['predictions', 'contour']

    batch_size = 256  # TODO: make this dynamic
    envs = sorted(df['env_name'].unique())
    n_envs = len(envs)
    n_cols = min(4, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5))
    axes = np.array(axes).reshape(-1)

    if 'interpolated_df' in kw:
        interpolated_df = kw['interpolated_df']
    else:
        interpolated_df = pd.DataFrame(columns=df.columns)

    if 'extrapolated_df' in kw:
        extrapolated_df = kw['extrapolated_df']
    else:
        extrapolated_df = pd.DataFrame(columns=df.columns)

    if 'compute_optimal_df' in kw:
        compute_optimal_df = kw['compute_optimal_df']
    else:
        compute_optimal_df = pd.DataFrame(columns=df.columns)

    for df_ in [df, interpolated_df, extrapolated_df, compute_optimal_df]:
        df_.dropna(subset=[data_efficiency_key], inplace=True)
        df_['compute'] = (
            10 * batch_size * df_['utd'] * df_['critic_params'] * df_[data_efficiency_key]
        )
        if 'delta' in kw:
            if isinstance(kw['delta'], dict):
                df_['budget'] = (
                    df_['compute'] + df_['env_name'].map(kw['delta']) * df_[data_efficiency_key]
                )
            else:
                df_['budget'] = df_['compute'] + kw['delta'] * df_[data_efficiency_key]

    if mode.startswith('budget'):
        plot_key = 'budget'
    elif mode.startswith('compute'):
        plot_key = 'compute'
    else:
        plot_key = data_efficiency_key
    cmap = colormaps['viridis']

    if mode.endswith('contour'):
        optima = []

    global_xs = sorted(
        set(df['utd'].values)
        | set(interpolated_df['utd'].values)
        | set(extrapolated_df['utd'].values)
        | set(compute_optimal_df['utd'].values)
    )
    global_ys = sorted(
        set(df['critic_params'].values)
        | set(interpolated_df['critic_params'].values)
        | set(extrapolated_df['critic_params'].values)
        | set(compute_optimal_df['critic_params'].values)
    )

    for i, (env, ax) in enumerate(zip(envs, axes)):
        env_data = df[df['env_name'] == env]
        interpolated_env_data = interpolated_df[interpolated_df['env_name'] == env]
        extrapolated_env_data = extrapolated_df[extrapolated_df['env_name'] == env]
        compute_optimal_env_data = compute_optimal_df[compute_optimal_df['env_name'] == env]
        x = env_data['utd']
        y = env_data['critic_params']
        z = env_data[plot_key]
        interpolated_x = interpolated_env_data['utd']
        interpolated_y = interpolated_env_data['critic_params']
        interpolated_z = interpolated_env_data[plot_key]
        extrapolated_x = extrapolated_env_data['utd']
        extrapolated_y = extrapolated_env_data['critic_params']
        extrapolated_z = extrapolated_env_data[plot_key]
        compute_optimal_x = compute_optimal_env_data['utd']
        compute_optimal_y = compute_optimal_env_data['critic_params']
        compute_optimal_z = compute_optimal_env_data[plot_key]
        env_optima = []

        all_xs = sorted(set(x.values) | set(extrapolated_x.values) | set(compute_optimal_x.values))
        all_ys = sorted(set(y.values) | set(extrapolated_y.values) | set(compute_optimal_y.values))

        # Used for contours
        zmin = max(min(z), 1e-8)
        zmax = max(z)

        if mode.startswith('budget') and isinstance(kw['delta'], dict):
            ax.set_title(f'{env}, $\delta={kw["delta"][env]:.0e}$')
        else:
            ax.set_title(env)
        ax.set_xlabel('utd')
        ax.set_ylabel('critic_params')
        ax.set_yscale('log')
        ax.set_xscale('log')

        def make_d_predictions(xmin, xmax, ymin, ymax):
            utd_smooth = np.logspace(np.log10(xmin), np.log10(xmax), 200)
            critic_params_smooth = np.logspace(np.log10(ymin), np.log10(ymax), 200)
            x_smooth, y_smooth = np.meshgrid(utd_smooth, critic_params_smooth)
            smooth_df = pd.DataFrame(
                {'utd': x_smooth.flatten(), 'critic_params': y_smooth.flatten()}
            )
            smooth_df['env_name'] = env
            smooth_predictions = predict_fn(smooth_df)
            smooth_predictions = np.array(smooth_predictions).reshape(
                len(utd_smooth), len(critic_params_smooth)
            )
            return x_smooth, y_smooth, smooth_predictions

        def make_c_predictions(xmin, xmax, ymin, ymax):
            x_smooth, y_smooth, smooth_d_predictions = make_d_predictions(xmin, xmax, ymin, ymax)
            smooth_computes = 10 * batch_size * x_smooth * y_smooth * smooth_d_predictions
            return x_smooth, y_smooth, smooth_computes

        def make_budget_predictions(delta, xmin, xmax, ymin, ymax):
            x_smooth, y_smooth, smooth_c_predictions = make_c_predictions(xmin, xmax, ymin, ymax)
            _, _, smooth_d_predictions = make_d_predictions(xmin, xmax, ymin, ymax)
            return x_smooth, y_smooth, smooth_c_predictions + delta * smooth_d_predictions

        if mode.startswith('data'):
            grid_predict_fn = make_d_predictions
        elif mode.startswith('compute'):
            grid_predict_fn = make_c_predictions
        elif mode.startswith('budget'):
            assert 'delta' in kw
            if isinstance(kw['delta'], dict):
                grid_predict_fn = partial(make_budget_predictions, kw['delta'][env])
            else:
                grid_predict_fn = partial(make_budget_predictions, kw['delta'])
        else:
            raise ValueError

        # Predictions with larger range and set color range
        xmin, xmax = expand_log_range(
            min(global_xs), max(global_xs), amount=0.5 if mode.startswith('budget') else 0.2
        )
        ymin, ymax = expand_log_range(
            min(global_ys), max(global_ys), amount=0.5 if mode.startswith('budget') else 0.2
        )
        x_smooth_plot, y_smooth_plot, smooth_predictions = grid_predict_fn(xmin, xmax, ymin, ymax)
        vmin = min(zmin, smooth_predictions.min())
        vmax = max(zmax, smooth_predictions.max())
        norm = LogNorm(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        if mode.endswith('predictions'):
            # x_in_range_idx = (x_smooth_plot[0] >= xmin) & (x_smooth_plot[0] <= xmax)
            # y_in_range_idx = (y_smooth_plot[:, 0] >= ymin) & (y_smooth_plot[:, 0] <= ymax)
            # plot_slice = np.ix_(x_in_range_idx, y_in_range_idx)

            ax.pcolormesh(
                # x_smooth_plot[plot_slice],
                # y_smooth_plot[plot_slice],
                # smooth_predictions[plot_slice],
                x_smooth_plot,
                y_smooth_plot,
                smooth_predictions,
                norm=norm,
                cmap=cmap,
                shading='auto',
            )
            # ax.set_autoscale_on(False)

            # def add_scatter_with_errors(x_vals, y_vals, z_vals, marker='o', size=100):
            #     ax.scatter(x_vals, y_vals, c=cmap(norm(z_vals)), s=size, marker=marker, edgecolor='white', zorder=10)
            #     input_df = pd.DataFrame({'utd': x_vals, 'critic_params': y_vals, 'env_name': env})
            #     predictions = predict_fn(input_df)
            #     for xi, yi, zi, pred in zip(x_vals, y_vals, z_vals, predictions):
            #         # pct_error = 100 * (np.log(zi) - np.log(pred)) / np.log(pred)
            #         pct_error = 100 * (pred - zi) / zi
            #         label = ('+' if pct_error > 0 else '') + f'{pct_error:.1f}'
            #         ax.text(xi, yi, label, color='white', ha='center', va='center', zorder=20, fontsize=8)

            # add_scatter_with_errors(x, y, z, size=500)
            # if len(interpolated_x) > 0:
            #     add_scatter_with_errors(interpolated_x, interpolated_y, interpolated_z, marker='s', size=500)
            # if len(extrapolated_x) > 0:
            #     add_scatter_with_errors(extrapolated_x, extrapolated_y, extrapolated_z, marker='D', size=500)
            # if len(compute_optimal_x) > 0:
            #     add_scatter_with_errors(compute_optimal_x, compute_optimal_y, compute_optimal_z, marker='*', size=500)

        elif mode.endswith('contour'):
            # x_in_range_idx = (x_smooth_plot[0] >= xmin) & (x_smooth_plot[0] <= xmax)
            # y_in_range_idx = (y_smooth_plot[:, 0] >= ymin) & (y_smooth_plot[:, 0] <= ymax)
            # plot_slice = np.ix_(x_in_range_idx, y_in_range_idx)

            if mode == 'data_contour':
                objective_fn = c_cost
            else:
                objective_fn = d_cost

            num_levels = 10
            iso_levels = np.logspace(np.log10(zmin), np.log10(zmax), num_levels)
            contour = ax.contour(
                # x_smooth_plot[plot_slice],
                # y_smooth_plot[plot_slice],
                # smooth_predictions[plot_slice],
                x_smooth_plot,
                y_smooth_plot,
                smooth_predictions,
                levels=iso_levels,
                norm=norm,
                cmap=cmap,
            )

            if mode.startswith('budget'):
                min_idx = smooth_predictions.flatten().argmin()
                opt_utd, opt_critic_params = (
                    x_smooth_plot.flatten()[min_idx],
                    y_smooth_plot.flatten()[min_idx],
                )
                opt_cost = smooth_predictions.flatten()[min_idx]
                data_efficiency = predict_fn(
                    pd.DataFrame(
                        {'utd': [opt_utd], 'critic_params': [opt_critic_params], 'env_name': [env]}
                    )
                )
                optima.append((env, opt_cost, opt_utd, opt_critic_params, data_efficiency[0]))
                ax.scatter(
                    opt_utd,
                    opt_critic_params,
                    c=cmap(norm(opt_cost)),
                    s=250,
                    marker='*',
                    edgecolor='black',
                    zorder=100,
                )

            else:
                smooth_costs = objective_fn(env, x_smooth_plot, y_smooth_plot, predict_fn)

                for collection, level in zip(contour.collections, iso_levels):
                    contour_colors = collection.get_edgecolor()
                    paths = collection.get_paths()
                    if not paths:
                        continue
                    assert len(paths) == len(contour_colors) == 1
                    path = paths[0]
                    contour_color = contour_colors[0]
                    verts = path.vertices  # (n_points, 2)
                    cost_utds, cost_critic_params = verts[:, 0], verts[:, 1]
                    costs = objective_fn(
                        env, cost_utds, cost_critic_params, predict_fn
                    )  # (n_points,)
                    min_idx = np.nanargmin(costs)
                    if min_idx not in {0, len(costs) - 1}:
                        opt_utd, opt_critic_params, cost_opt = (
                            cost_utds[min_idx],
                            cost_critic_params[min_idx],
                            costs[min_idx],
                        )
                        optima.append((env, level, opt_utd, opt_critic_params, cost_opt))
                        env_optima.append((opt_utd, opt_critic_params))
                        # ax.plot(
                        #     opt_utd,
                        #     opt_critic_params,
                        #     marker='*',
                        #     color=contour_color,
                        #     markersize=10,
                        #     zorder=5,
                        #     mec='white',
                        # )
                        ax.contour(
                            # x_smooth_plot[plot_slice],
                            # y_smooth_plot[plot_slice],
                            # smooth_costs[plot_slice],
                            x_smooth_plot,
                            y_smooth_plot,
                            smooth_costs,
                            levels=[cost_opt],
                            colors='gray',
                            alpha=0.3,
                            zorder=-10,
                        )

                if len(env_optima) > 0:
                    env_opt_utd, env_opt_critic_params = zip(*env_optima)
                    optima_slope, optima_intercept = np.polyfit(
                        np.log(env_opt_utd), np.log(env_opt_critic_params), 1
                    )
                    smooth_x = np.linspace(xmin, xmax, 100)
                    smooth_y = np.exp(optima_slope * np.log(smooth_x) + optima_intercept)
                    ax.plot(smooth_x, smooth_y, color='gray', linestyle='--', alpha=0.5)

        ax.scatter(x, y, c=cmap(norm(z)), s=100, marker='o', edgecolor='white', zorder=10)
        if len(interpolated_x) > 0:
            ax.scatter(
                interpolated_x,
                interpolated_y,
                c=cmap(norm(interpolated_z)),
                s=100,
                marker='s',
                edgecolor='white',
                zorder=10,
            )
        if len(extrapolated_x) > 0:
            ax.scatter(
                extrapolated_x,
                extrapolated_y,
                c=cmap(norm(extrapolated_z)),
                s=100,
                marker='D',
                edgecolor='white',
                zorder=10,
            )
        if len(compute_optimal_x) > 0:
            ax.scatter(
                compute_optimal_x,
                compute_optimal_y,
                c=cmap(norm(compute_optimal_z)),
                s=100,
                marker='*',
                edgecolor='white',
                zorder=10,
            )

        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(all_xs)
        ax.set_xticklabels([int(x) for x in all_xs])
        ax.set_ylim(ymin, ymax)

        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(all_ys)
        ax.set_yticklabels([abbreviate_number(y) for y in all_ys])
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(sm, ax=ax)

    for i in range(len(envs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if mode.startswith('budget') and not isinstance(kw['delta'], dict):
        plt.suptitle(f'{plot_key}, $\delta={kw["delta"]:.0e}$', y=1.05)
    else:
        plt.suptitle(plot_key, y=1.05)
    plt.show()

    if mode == 'data_contour':
        return pd.DataFrame(
            optima,
            columns=[
                'env_name',
                'iso_data_to_thresh',
                'opt_utd',
                'opt_critic_params',
                'opt_compute',
            ],
        )
    elif mode == 'compute_contour':
        return pd.DataFrame(
            optima,
            columns=[
                'env_name',
                'iso_compute',
                'opt_utd',
                'opt_critic_params',
                'opt_data_efficiency',
            ],
        )
    elif mode == 'budget_contour':
        return pd.DataFrame(
            optima,
            columns=['env_name', 'opt_budget', 'opt_utd', 'opt_critic_params', 'data_efficiency'],
        )


def plot_optimal_hparams_scatter_pretty(df, predict_fn, data_efficiency_key: str, mode: str, **kw):
    assert mode.split('_')[0] in ['data', 'compute', 'budget']
    assert mode.split('_')[1] in ['predictions', 'contour']
    original_rc_params = plt.rcParams.copy()
    qscaled_plot_utils.set_theme()

    batch_size = 256  # TODO: make this dynamic
    envs = sorted(df['env_name'].unique())
    n_envs = len(envs)
    n_rows = 2 if n_envs >= 4 else 1
    n_cols = int(np.ceil(n_envs / n_rows))

    if set(envs) == {
        'DMC-hard',
        'DMC-medium',
        'h1-crawl-v0',
        'h1-pole-v0',
        'h1-stand-v0',
        'humanoid-stand',
    }:
        envs = [
            'DMC-medium',
            'DMC-hard',
            'humanoid-stand',
            'h1-crawl-v0',
            'h1-pole-v0',
            'h1-stand-v0',
        ]

    if set(envs) == {
        'DMC-hard',
        'DMC-medium',
        'h1-crawl-v0',
        'h1-pole-v0',
        'h1-stand-v0',
        'humanoid-stand',
        'h1-crawl-v0simba',
        'h1-stand-v0simba',
    }:
        envs = [
            'DMC-medium',
            'DMC-hard',
            'h1-crawl-v0',
            'h1-crawl-v0simba',
            'humanoid-stand',
            'h1-pole-v0',
            'h1-stand-v0',
            'h1-stand-v0simba',
        ]

    col_width = kw.get('col_width', 4)
    row_height = kw.get('row_height', 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * col_width, n_rows * row_height))
    axes = np.array(axes).reshape(-1)

    if 'interpolated_df' in kw:
        interpolated_df = kw['interpolated_df']
    else:
        interpolated_df = pd.DataFrame(columns=df.columns)

    if 'extrapolated_df' in kw:
        extrapolated_df = kw['extrapolated_df']
    else:
        extrapolated_df = pd.DataFrame(columns=df.columns)

    if 'compute_optimal_df' in kw:
        compute_optimal_df = kw['compute_optimal_df']
    else:
        compute_optimal_df = pd.DataFrame(columns=df.columns)

    for df_ in [df, interpolated_df, extrapolated_df, compute_optimal_df]:
        df_.dropna(subset=[data_efficiency_key], inplace=True)
        df_['compute'] = (
            10 * batch_size * df_['utd'] * df_['critic_params'] * df_[data_efficiency_key]
        )
        if 'delta' in kw:
            if isinstance(kw['delta'], dict):
                df_['budget'] = (
                    df_['compute']
                    + df_['env_name'].map(kw['delta']).astype(float) * df_[data_efficiency_key]
                )
            else:
                df_['budget'] = df_['compute'] + float(kw['delta']) * df_[data_efficiency_key]

    if mode.startswith('budget'):
        plot_key = 'budget'
    elif mode.startswith('compute'):
        plot_key = 'compute'
    else:
        plot_key = data_efficiency_key

    if kw.get('color_scheme', 'qscaled') != 'qscaled':
        cmap = colormaps[kw['color_scheme']]
    else:
        cmap = LinearSegmentedColormap.from_list(
            'custom_gradient', [qscaled_plot_utils.COLORS[0], qscaled_plot_utils.COLORS[1]]
        )

    if mode.endswith('contour'):
        optima = []

    global_xs = sorted(
        set(df['utd'].values)
        | set(interpolated_df['utd'].values)
        | set(extrapolated_df['utd'].values)
        | set(compute_optimal_df['utd'].values)
    )
    global_ys = sorted(
        set(df['critic_params'].values)
        | set(interpolated_df['critic_params'].values)
        | set(extrapolated_df['critic_params'].values)
        | set(compute_optimal_df['critic_params'].values)
    )

    final_expand_amt = 0.2
    # xmin, xmax = expand_log_range(min(global_xs), max(global_xs), amount=final_expand_amt)
    # ymin, ymax = expand_log_range(min(global_ys), max(global_ys), amount=final_expand_amt)

    state = {}

    def make_d_predictions(env, xmin, xmax, ymin, ymax, granularity=200):
        utd_smooth = np.logspace(np.log10(xmin), np.log10(xmax), granularity)
        critic_params_smooth = np.logspace(np.log10(ymin), np.log10(ymax), granularity)
        x_smooth, y_smooth = np.meshgrid(utd_smooth, critic_params_smooth)
        smooth_df = pd.DataFrame({'utd': x_smooth.flatten(), 'critic_params': y_smooth.flatten()})
        smooth_df['env_name'] = env
        smooth_predictions = predict_fn(smooth_df)
        smooth_predictions = np.array(smooth_predictions).reshape(
            len(utd_smooth), len(critic_params_smooth)
        )
        return x_smooth, y_smooth, smooth_predictions

    def make_c_predictions(env, xmin, xmax, ymin, ymax, granularity=200):
        x_smooth, y_smooth, smooth_d_predictions = make_d_predictions(
            env, xmin, xmax, ymin, ymax, granularity
        )
        smooth_computes = 10 * batch_size * x_smooth * y_smooth * smooth_d_predictions
        return x_smooth, y_smooth, smooth_computes

    def make_budget_predictions(delta, env, xmin, xmax, ymin, ymax, granularity=200):
        x_smooth, y_smooth, smooth_c_predictions = make_c_predictions(
            env, xmin, xmax, ymin, ymax, granularity
        )
        _, _, smooth_d_predictions = make_d_predictions(env, xmin, xmax, ymin, ymax, granularity)
        return x_smooth, y_smooth, smooth_c_predictions + delta * smooth_d_predictions

    for i, (env, ax) in enumerate(zip(envs, axes)):
        env_data = df[df['env_name'] == env]
        x = env_data['utd']
        y = env_data['critic_params']
        z = env_data[plot_key]

        extrapolated_env_data = extrapolated_df[extrapolated_df['env_name'] == env]
        x_extrapolated = extrapolated_env_data['utd']
        y_extrapolated = extrapolated_env_data['critic_params']
        z_extrapolated = extrapolated_env_data[plot_key]

        all_xs = sorted(set(x.values) | set(x_extrapolated.values))
        all_ys = sorted(set(y.values) | set(y_extrapolated.values))

        # Used for contours
        zmin = max(min(z), 1e-8)
        zmax = max(z)

        if not extrapolated_env_data.empty:
            zmin = min(zmin, min(z_extrapolated))
            zmax = max(zmax, max(z_extrapolated))

        if mode.startswith('budget') and isinstance(kw['delta'], dict):
            ax.set_title(f'{env} ($\delta$={kw["delta"][env]})', fontsize='xx-large', y=1.05)
        else:
            env_fmt = env
            if any(env.endswith('simba') for env in envs):
                if 'simba' in env:
                    env_fmt = env.replace('simba', '') + ', SimbaV2'
                else:
                    env_fmt = env + ', BRO'
            ax.set_title(env_fmt, fontsize='xx-large', y=1.05)
        ax.set_xlabel('utd')
        ax.set_ylabel('critic_params')
        ax.set_yscale('log')
        ax.set_xscale('log')

        if mode.startswith('data'):
            grid_predict_fn = partial(make_d_predictions, env)
        elif mode.startswith('compute'):
            grid_predict_fn = partial(make_c_predictions, env)
        elif mode.startswith('budget'):
            assert 'delta' in kw
            if isinstance(kw['delta'], dict):
                grid_predict_fn = partial(make_budget_predictions, float(kw['delta'][env]), env)
            else:
                grid_predict_fn = partial(make_budget_predictions, float(kw['delta']), env)
        else:
            raise ValueError

        # Set color range
        # _, _, range_predictions = grid_predict_fn(xmin, xmax, ymin, ymax, granularity=2)
        _, _, range_predictions = grid_predict_fn(min(x), max(x), min(y), max(y), granularity=2)
        vmin = min(zmin, range_predictions.min())
        vmax = max(zmax, range_predictions.max())

        state[env] = {
            'x': x,
            'y': y,
            'z': z,
            'all_xs': all_xs,
            'all_ys': all_ys,
            'zmin': zmin,
            'zmax': zmax,
            'vmin': zmin,  # vmin,
            'vmax': zmax,  # vmax,
            'grid_predict_fn': grid_predict_fn,
        }

    vmin = min(state[env]['vmin'] for env in envs)
    vmax = max(state[env]['vmax'] for env in envs)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # expand to draw gray isocontours
    xmin, xmax = expand_log_range(min(global_xs), max(global_xs), amount=2.0)
    ymin, ymax = expand_log_range(min(global_ys), max(global_ys), amount=2.0)
    if mode.startswith('data'):
        cbar_label = r'$\mathcal{D}_J$: Data until $J_{\text{env}}$'
    elif mode.startswith('compute'):
        cbar_label = r'$\mathcal{C}_J$: Compute until $J_{\text{env}}$'
    elif mode.startswith('budget'):
        cbar_label = r'$\mathcal{F}_J$: Budget until $J_{\text{env}}$'

    for i, (env, ax) in enumerate(zip(envs, axes)):
        if i in kw.get('remove_axes', []):
            continue

        i_ = i % n_cols
        j_ = i // n_cols

        x = state[env]['x']
        y = state[env]['y']
        z = state[env]['z']
        all_xs = state[env]['all_xs']
        all_ys = state[env]['all_ys']
        zmin = state[env]['zmin']
        zmax = state[env]['zmax']
        grid_predict_fn = state[env]['grid_predict_fn']
        xmin, xmax = expand_log_range(min(all_xs), max(all_xs), amount=final_expand_amt)
        ymin, ymax = expand_log_range(min(all_ys), max(all_ys), amount=final_expand_amt)

        # actual predictions
        x_smooth_plot, y_smooth_plot, smooth_predictions = grid_predict_fn(xmin, xmax, ymin, ymax)

        env_optima = []

        if mode.endswith('predictions'):
            ax.pcolormesh(
                y_smooth_plot,
                x_smooth_plot,
                smooth_predictions,
                norm=norm,
                cmap=cmap,
                shading='auto',
            )

        elif mode.endswith('contour'):
            if mode == 'data_contour':
                objective_fn = c_cost
            else:
                objective_fn = d_cost

            num_levels = kw.get('num_levels', 10)
            # iso_levels = np.logspace(np.log10(vmin), np.log10(vmax), num_levels)
            iso_levels = np.logspace(np.log10(zmin), np.log10(zmax), num_levels)
            plot_first_k_iso_levels = kw.get('plot_first_k_iso_levels', num_levels)
            iso_levels = iso_levels[-plot_first_k_iso_levels:]
            contour = ax.contour(
                y_smooth_plot,
                x_smooth_plot,
                smooth_predictions,
                levels=iso_levels,
                norm=norm,
                cmap=cmap,
                linewidths=[3] * plot_first_k_iso_levels
                + [0] * (num_levels - plot_first_k_iso_levels),
            )

            if mode.startswith('budget'):
                min_idx = smooth_predictions.flatten().argmin()
                opt_utd, opt_critic_params = (
                    x_smooth_plot.flatten()[min_idx],
                    y_smooth_plot.flatten()[min_idx],
                )
                opt_cost = smooth_predictions.flatten()[min_idx]
                data_efficiency = predict_fn(
                    pd.DataFrame(
                        {'utd': [opt_utd], 'critic_params': [opt_critic_params], 'env_name': [env]}
                    )
                )
                optima.append((env, opt_cost, opt_utd, opt_critic_params, data_efficiency[0]))
                ax.scatter(
                    opt_critic_params,
                    opt_utd,
                    color=cmap(norm(opt_cost)),
                    s=250,
                    marker='*',
                    edgecolor='black',
                    zorder=100,
                )

                # if 'raw_params' in kw:
                #     raw_params = kw['raw_params']

                #     ax.scatter(raw_params[env]['utd'], raw_params[env]['critic_params'], c=cmap(norm(raw_params[env]['cost'])), s=200, marker='*', linewidth=3, zorder=100)

            else:
                smooth_costs = objective_fn(env, x_smooth_plot, y_smooth_plot, predict_fn)
                cnt = 0
                for collection, level in zip(contour.collections, iso_levels):
                    if cnt >= plot_first_k_iso_levels:
                        break
                    cnt += 1
                    # for i, (collection, level) in enumerate(zip(contour.collections, iso_levels)):  # crazy matplotlib bug wtf???
                    contour_colors = collection.get_edgecolor()
                    paths = collection.get_paths()
                    if not paths:
                        continue
                    assert len(paths) == len(contour_colors) == 1
                    path = paths[0]
                    contour_color = contour_colors[0]
                    verts = path.vertices  # (n_points, 2)
                    cost_critic_params, cost_utds = verts[:, 0], verts[:, 1]
                    costs = objective_fn(
                        env, cost_utds, cost_critic_params, predict_fn
                    )  # (n_points,)
                    min_idx = np.nanargmin(costs)
                    if min_idx not in {0, len(costs) - 1}:
                        opt_utd, opt_critic_params, cost_opt = (
                            cost_utds[min_idx],
                            cost_critic_params[min_idx],
                            costs[min_idx],
                        )
                        optima.append((env, level, opt_utd, opt_critic_params, cost_opt))
                        env_optima.append((opt_utd, opt_critic_params))
                        if kw.get('show_cost_contour', True):
                            ax.contour(
                                y_smooth_plot,
                                x_smooth_plot,
                                smooth_costs,
                                levels=[cost_opt],
                                colors='gray',
                                alpha=0.3,
                                linewidths=2,
                                zorder=-10,
                            )
                        if kw.get('show_optima_as_stars', False):
                            if kw.get('params') is not None and mode.startswith('data'):
                                # verify math works
                                a, alpha, b, beta, d_min = kw['params'][env][:5]
                                a *= d_min ** (1 / alpha)
                                b *= d_min ** (1 / beta)
                                opt_utd = a * ((1 + alpha / beta) / (level - d_min)) ** (1 / alpha)
                                opt_critic_params = b * ((1 + beta / alpha) / (level - d_min)) ** (
                                    1 / beta
                                )
                            ax.scatter(
                                opt_critic_params,
                                opt_utd,
                                color='gold',
                                s=280,
                                marker='*',
                                edgecolor='black',
                                zorder=100,
                            )
                            # ax.scatter(opt_utd, opt_critic_params, color='gold', s=320, marker='*', zorder=99)

                if len(env_optima) > 0:
                    env_opt_utd, env_opt_critic_params = zip(*env_optima)
                    optima_slope, optima_intercept = np.polyfit(
                        np.log(env_opt_utd), np.log(env_opt_critic_params), 1
                    )
                    smooth_x = np.linspace(xmin, xmax, 100)
                    smooth_y = np.exp(optima_slope * np.log(smooth_x) + optima_intercept)
                    if kw.get('show_optima_as_line', True):
                        ax.plot(
                            smooth_y,
                            smooth_x,
                            color='gray',
                            linestyle='--',
                            alpha=0.8,
                            linewidth=3,
                            label='Compute-optimal' if mode.startswith('data') else 'Data-optimal',
                            zorder=kw.get('line_zorder', 50),
                        )

        ax.scatter(
            y,
            x,
            c=cmap(norm(z)),
            s=150,
            marker='o',
            edgecolor='black' if kw.get('color_scheme', 'qscaled') == 'qscaled' else 'white',
            zorder=10,
        )

        if mode.endswith('predictions'):
            ax.scatter(
                y_extrapolated,
                x_extrapolated,
                c=cmap(norm(z_extrapolated)),
                s=150,
                marker='X',
                edgecolor='black' if kw.get('color_scheme', 'qscaled') == 'qscaled' else 'white',
                zorder=10,
            )

        rliable_plot_utils._annotate_and_decorate_axis(
            ax,
            xlabel=r'$N$: model size' if j_ == n_rows - 1 else '',
            ylabel=r'$\sigma$: UTD' if i_ == 0 else '',
            labelsize='xx-large',
            ticklabelsize='xx-large',
            grid_alpha=0.2,
            legend=False,
        )

        ax.yaxis.set_minor_locator(ticker.NullLocator())
        # ax.set_xlim(xmin, xmax)
        ax.set_yticks(all_xs)
        ax.set_yticklabels([int(x) for x in all_xs])
        ax.set_xlim(ymin, ymax)

        ax.xaxis.set_minor_locator(ticker.NullLocator())
        # ax.set_ylim(ymin, ymax)
        # if i == 0:
        if True:

            def format_to_m(value, decimals=1, strip_leading_zero=True):
                """Format number as millions, e.g. 183000 -> '.2M'."""
                scaled = round(value / 1e6, decimals)  # scale to millions and round
                s = f'{scaled:.{decimals}f}'  # fixed decimals
                if strip_leading_zero:
                    if s.startswith('0'):  # '.x' instead of '0.x'
                        s = s[1:]
                    elif s.startswith('-0'):  # '-.x' instead of '-0.x'
                        s = '-' + s[2:]
                return s + 'M'

            ax.set_xticks(all_ys)
            ax.set_xticklabels(
                [format_to_m(x) if x < 1e6 else abbreviate_number(x) for x in all_ys]
            )
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # ax.set_aspect('equal', adjustable='box')

        if i_ == 0 and not mode.startswith('budget') and kw.get('show_cost_contour', True):
            handles, labels = ax.get_legend_handles_labels()
            iso_label = 'Iso-compute lines' if mode.startswith('data') else 'Iso-data lines'
            iso_line = plt.Line2D([], [], color='gray', alpha=0.3, linewidth=2, label=iso_label)
            handles.insert(0, iso_line)
            labels.insert(0, iso_label)
            # ax.legend(handles=handles, labels=labels, prop={'size': 14}, ncol=1, frameon=False, loc='lower left')
            if kw.get('show_legend', True):
                ax.legend(
                    handles=handles,
                    labels=labels,
                    prop={'size': 14},
                    ncol=1,
                    frameon=True,
                    loc='lower left',
                    framealpha=0.5,
                    edgecolor='white',
                )

        xmin, xmax = expand_log_range(min(all_xs), max(all_xs), amount=final_expand_amt)
        ymin, ymax = expand_log_range(min(all_ys), max(all_ys), amount=final_expand_amt)
        ax.set_xlim(ymin, ymax)
        ax.set_ylim(xmin, xmax)

        # if i == len(envs) - 1:
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes('right', size='10%', pad=0.5)
        #     cbar = plt.colorbar(sm, cax=cax)
        #     cbar.set_label(cbar_label, size='xx-large')
        #     cbar.ax.tick_params(labelsize='xx-large')

    show_cbar = kw.get('show_cbar', True)
    use_cbar_spacing = kw.get('cbar_spacing', show_cbar)

    if use_cbar_spacing:
        plt.tight_layout(rect=[0, 0, 0.9, 1], w_pad=1.5, h_pad=1.5)
        cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])  # Adjust as needed
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, size='xx-large', color='black' if show_cbar else 'white')
        cbar.ax.tick_params(labelsize='xx-large')

        if 'cbar_scale' in kw:
            cbar.ax.text(
                1.05,
                1,
                f'×{kw["cbar_scale"]}',
                transform=cbar.ax.transAxes,
                ha='left',
                va='top',
                fontsize='x-large',
                alpha=0.8,
            )
            cbar_scale = float(kw['cbar_scale'])
        else:
            cbar_scale = 1

        if 'cbar_ticks' in kw:
            tick_locs = kw['cbar_ticks']
            tick_labels = [f'{round(tick / cbar_scale)}' for tick in tick_locs]
            cax.yaxis.set_major_locator(plt.FixedLocator(tick_locs))
            cax.yaxis.set_minor_locator(plt.NullLocator())
            cbar.set_ticks(tick_locs, labels=tick_labels, fontsize='xx-large')
            # cbar.ax.set_yticklabels(tick_labels, fontsize='xx-large')

        if not show_cbar:
            rect = patches.Rectangle(
                (0.9, 0),
                0.1,
                1,
                transform=fig.transFigure,
                facecolor='white',
                edgecolor='none',
                zorder=10000,
            )
            fig.patches.append(rect)

    else:
        plt.tight_layout(rect=[0, 0, 1, 1])

    # xmin, xmax = expand_log_range(min(global_xs), max(global_xs), amount=final_expand_amt)
    # ymin, ymax = expand_log_range(min(global_ys), max(global_ys), amount=final_expand_amt)
    # for ax in axes:
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)

    if kw.get('print_axes_locs', False):
        for i, ax in enumerate(axes):
            print(f'{i}: {ax.get_position()}')

    for i in range(len(envs), len(axes)):
        fig.delaxes(axes[i])

    for i in kw.get('remove_axes', []):
        axes[i].set_title('')
        axes[i].axis('off')
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=axes[i].transAxes,
            facecolor='white',
            edgecolor='none',
            zorder=10000,
        )
        axes[i].add_patch(rect)

    # plt.subplots_adjust(wspace=0.5)
    # fig.subplots_adjust(right=0.85)
    # if mode.startswith('budget') and not isinstance(kw['delta'], dict):
    #     plt.suptitle(f'{plot_key}, $\delta={kw["delta"]:.0e}$', y=1.05)
    # else:
    #     plt.suptitle(plot_key, y=1.05)

    if kw.get('save_path'):
        os.makedirs(os.path.dirname(kw['save_path']), exist_ok=True)
        plt.savefig(kw['save_path'], bbox_inches='tight')
    if kw.get('show_plot', True):
        plt.show()
    plt.close()

    plt.rcParams.update(original_rc_params)

    if mode == 'data_contour':
        return pd.DataFrame(
            optima,
            columns=[
                'env_name',
                'iso_data_to_thresh',
                'opt_utd',
                'opt_critic_params',
                'opt_compute',
            ],
        )
    elif mode == 'compute_contour':
        return pd.DataFrame(
            optima,
            columns=[
                'env_name',
                'iso_compute',
                'opt_utd',
                'opt_critic_params',
                'opt_data_efficiency',
            ],
        )
    elif mode == 'budget_contour':
        return pd.DataFrame(
            optima,
            columns=['env_name', 'opt_budget', 'opt_utd', 'opt_critic_params', 'data_efficiency'],
        )


def compute_optimal_slope_and_intercept(compute_optimal_hparams):
    def fit_log_log_slope_and_intercept(group):
        slope, intercept = np.polyfit(group['log_critic_params'], group['log_utd'], 1)
        eq_n_to_sigma = f'sigma = {np.exp(intercept):.4e} * N^{slope:.4f}'
        eq_sigma_to_n = f'N = {np.exp(intercept) ** (-1 / slope):.4e} * sigma^{1 / slope:.4f}'
        return pd.Series(
            {
                'slope': slope,
                'intercept': intercept,
                'eq_n_to_sigma': eq_n_to_sigma,
                'eq_sigma_to_n': eq_sigma_to_n,
            }
        )

    compute_optimal_hparams = deepcopy(compute_optimal_hparams)
    compute_optimal_hparams['log_critic_params'] = np.log(
        compute_optimal_hparams['opt_critic_params']
    )
    compute_optimal_hparams['log_utd'] = np.log(compute_optimal_hparams['opt_utd'])
    result = compute_optimal_hparams.groupby('env_name').apply(fit_log_log_slope_and_intercept)
    return result.reset_index()


def plot_metric_summary_vs_batch_size_group_utd(df, metric, summarize_how, filter_utd=None):
    df = deepcopy(df)
    if filter_utd is not None:
        df = df[df['utd'] == filter_utd]
    if summarize_how == 'end of training':
        if metric == 'return':
            df['summary'] = df[metric].apply(lambda x: x[-1])
        else:
            # Use np.nanmean to handle NaN values properly
            df['summary'] = df[metric].apply(lambda x: np.nanmean(x[-1, :]))
    elif summarize_how == 'mean over training':
        df['summary'] = df[metric].apply(lambda x: np.mean(x))
    elif summarize_how == 'median over training':
        df['summary'] = df[metric].apply(lambda x: np.median(x))
    elif summarize_how == 'last 10%':
        if metric == 'return':
            df['summary'] = df[metric].apply(lambda x: np.nanmean(x[-int(len(x) * 0.1) :]))
        else:
            df['summary'] = df[metric].apply(lambda x: np.nanmean(x[-int(len(x) * 0.1) :, :]))
    else:
        raise ValueError(f'Unknown summarize_how: {summarize_how}')

    envs = sorted(df['env_name'].unique())
    utds = sorted(df['utd'].unique())
    critic_widths = sorted(df['critic_width'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    # Use viridis color scheme but with more separation between colors
    color_sep = 4  # Number of colors to skip between each color, so that colors are more distinct
    colors = sns.color_palette('viridis', len(critic_widths) * color_sep)[::color_sep]
    # # Pick uniformly spaced colors from the entire viridis palette
    # if len(critic_widths) > 1:
    #     indices = np.linspace(0, 1, len(critic_widths))
    #     colors = [plt.cm.viridis(i) for i in indices]
    # else:
    #     colors = [plt.cm.viridis(0.5)]  # Use middle color if only one width
    fig, axes = plt.subplots(len(utds), len(envs), figsize=(len(envs) * 4, len(utds) * 3))
    axes = np.array(axes).reshape(len(utds), len(envs))

    lines, labels = [], []

    for i, utd in enumerate(utds):
        for j, env in enumerate(envs):
            ax = axes[i, j]
            if filter_utd is not None:
                ax.set_title(env)
            else:
                ax.set_title(f'{env}, $\sigma={utd}$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Batch Size')
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels(batch_sizes)

            slope_lines, slope_labels = [], []

            for critic_width, color in zip(critic_widths, colors):
                subset = df[
                    (df['env_name'] == env)
                    & (df['critic_width'] == critic_width)
                    & (df['utd'] == utd)
                ]
                if len(subset) == 0:
                    continue
                label = f'width={critic_width}'
                ax.plot(
                    subset['batch_size'],
                    subset['summary'],
                    'o',
                    label=label,
                    color=color,
                    alpha=0.3,  # More translucent points
                )
                # Use quadratic fit for critic widths 256 and 512, linear for others
                # if critic_width in [256, 512, 1024, 2048]:
                if len(subset['summary']) >= 4:
                    # Quadratic fit (polynomial of degree 2)
                    log_x = np.log(subset['batch_size'])
                    log_y = np.log(subset['summary'])

                    # Create polynomial features: [1, x, x²]
                    X = np.column_stack([np.ones(len(log_x)), log_x, log_x**2])

                    # Fit the model
                    quad_fit = sm.OLS(log_y, X).fit()

                    # Generate smooth curve for plotting
                    smooth_x = np.linspace(
                        np.log(subset['batch_size'].min()),
                        np.log(subset['batch_size'].max()),
                        100,
                    )

                    # Predict using the quadratic model
                    X_smooth = np.column_stack([np.ones(len(smooth_x)), smooth_x, smooth_x**2])
                    smooth_y = quad_fit.predict(X_smooth)

                    # Plot the curve
                    (line,) = ax.plot(
                        np.exp(smooth_x),
                        np.exp(smooth_y),
                        color=color,
                        linestyle='--',
                        alpha=1,
                        linewidth=2.5,
                    )

                    # Get quadratic coefficient for labeling
                    quad_coef = quad_fit.params[2]
                    r2 = quad_fit.rsquared
                    slope_lines.append(line)
                    slope_labels.append(f'quad={quad_coef:.2f}, R²={r2:.2f}')

                    # Find the minimum of the quadratic fit
                    if quad_coef > 0:  # Only if it's a U-shaped curve (positive quadratic term)
                        # For a quadratic y = ax² + bx + c, the minimum is at x = -b/(2a)
                        a, b = quad_fit.params[2], quad_fit.params[1]
                        min_x_log = -b / (2 * a) if a != 0 else None

                        # Check if minimum is within the data range
                        if (
                            min_x_log is not None
                            and min_x_log >= np.log(subset['batch_size'].min())
                            and min_x_log <= np.log(subset['batch_size'].max())
                        ):
                            min_x = np.exp(min_x_log)
                            min_y = np.exp(quad_fit.predict([1, min_x_log, min_x_log**2])[0])

                            # Plot X marker at the minimum
                            ax.plot(
                                min_x, min_y, 'x', color=color, markersize=10, markeredgewidth=3
                            )
                else:
                    # Linear fit for other widths
                    linear_fit = sm.OLS(
                        np.log(subset['summary']),
                        sm.add_constant(np.log(subset['batch_size'])),
                    ).fit()
                    smooth_x = np.linspace(
                        np.log(subset['batch_size'].min()),
                        np.log(subset['batch_size'].max()),
                        100,
                    )
                    smooth_y = linear_fit.predict(sm.add_constant(smooth_x))
                    (line,) = ax.plot(
                        np.exp(smooth_x),
                        np.exp(smooth_y),
                        color=color,
                        linestyle='--',
                        alpha=1,
                        linewidth=2.5,  # Thicker lines
                    )
                    slope = linear_fit.params.iloc[1]
                    r2 = linear_fit.rsquared
                    slope_lines.append(line)
                    slope_labels.append(f'slope={slope:.2f}, R²={r2:.2f}')

                if label not in labels:
                    labels.append(label)
                    lines.append(line)

            ax.legend(slope_lines, slope_labels, fontsize=8, frameon=False)

    sorted_idx = sorted(range(len(labels)), key=lambda i: int(labels[i].split('=')[1]))
    lines = [lines[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(labels))
    if filter_utd is None:
        fig.suptitle(f'{metric}, {summarize_how}', fontsize=16)
    else:
        fig.suptitle(f'{metric}, $\sigma$={filter_utd}, {summarize_how}', fontsize=16)
    plt.tight_layout()


def plot_metric_summary_vs_batch_size_group_width(df, metric, summarize_how, filter_width=None):
    df = deepcopy(df)
    if filter_width is not None:
        df = df[df['utd'] == filter_width]
    if summarize_how == 'end of training':
        df['summary'] = df[metric].apply(lambda x: x[~np.isnan(x)][-1])
    elif summarize_how == 'mean over training':
        df['summary'] = df[metric].apply(lambda x: np.mean(x))
    elif summarize_how == 'median over training':
        df['summary'] = df[metric].apply(lambda x: np.median(x))
    else:
        raise ValueError(f'Unknown summarize_how: {summarize_how}')

    envs = sorted(df['env_name'].unique())
    utds = sorted(df['utd'].unique())
    critic_widths = sorted(df['critic_width'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    # Pick uniformly spaced colors from the entire viridis palette
    if len(utds) > 1:
        indices = np.linspace(0, 1, len(utds))
        colors = [plt.cm.viridis(i) for i in indices]
    else:
        colors = [plt.cm.viridis(0.5)]  # Use middle color if only one UTD
    fig, axes = plt.subplots(
        len(critic_widths), len(envs), figsize=(len(envs) * 4, len(critic_widths) * 3)
    )
    axes = np.array(axes).reshape(len(critic_widths), len(envs))

    lines, labels = [], []

    for i, critic_width in enumerate(critic_widths):
        for j, env in enumerate(envs):
            ax = axes[i, j]
            if filter_width is not None:
                ax.set_title(env)
            else:
                ax.set_title(f'{env}, width={critic_width}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Batch Size')
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels(batch_sizes)

            slope_lines, slope_labels = [], []

            for utd, color in zip(utds, colors):
                subset = df[
                    (df['env_name'] == env)
                    & (df['critic_width'] == critic_width)
                    & (df['utd'] == utd)
                ]
                if len(subset) == 0:
                    continue
                label = f'$\sigma$={utd}'
                (line,) = ax.plot(
                    subset['batch_size'],
                    subset['summary'],
                    'o',
                    label=label,
                    color=color,
                    alpha=0.3,  # More translucent points
                )
                linear_fit = sm.OLS(
                    np.log(subset['summary']),
                    sm.add_constant(np.log(subset['batch_size'])),
                ).fit()
                smooth_x = np.linspace(
                    np.log(subset['batch_size'].min()),
                    np.log(subset['batch_size'].max()),
                    100,
                )
                smooth_y = linear_fit.predict(sm.add_constant(smooth_x))
                (line,) = ax.plot(
                    np.exp(smooth_x),
                    np.exp(smooth_y),
                    color=color,
                    linestyle='--',
                    alpha=1,
                    linewidth=2.5,  # Thicker lines
                )

                slope = linear_fit.params.iloc[1]
                r2 = linear_fit.rsquared
                slope_lines.append(line)
                slope_labels.append(f'slope={slope:.2f}, R²={r2:.2f}')

                if label not in labels:
                    labels.append(label)
                    lines.append(line)

            ax.legend(slope_lines, slope_labels, fontsize=8, frameon=False)

    sorted_idx = sorted(range(len(labels)), key=lambda i: float(labels[i].split('=')[1]))
    lines = [lines[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(labels))
    if filter_width is None:
        fig.suptitle(f'{metric}, {summarize_how}', fontsize=16)
    else:
        fig.suptitle(f'{metric}, width={filter_width}, {summarize_how}', fontsize=16)
    plt.tight_layout()


def plot_metric_summary_vs_width_group_batch_size(df, metric, summarize_how, filter_bs=None):
    df = deepcopy(df)
    if filter_bs is not None:
        df = df[df['batch_size'] == filter_bs]
    if summarize_how == 'end of training':
        df['summary'] = df[metric].apply(lambda x: x[~np.isnan(x)][-1])
    elif summarize_how == 'mean over training':
        df['summary'] = df[metric].apply(lambda x: np.mean(x))
    elif summarize_how == 'median over training':
        df['summary'] = df[metric].apply(lambda x: np.median(x))
    else:
        raise ValueError(f'Unknown summarize_how: {summarize_how}')

    envs = sorted(df['env_name'].unique())
    utds = sorted(df['utd'].unique())
    critic_widths = sorted(df['critic_width'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    # Pick uniformly spaced colors from the entire viridis palette
    if len(utds) > 1:
        indices = np.linspace(0, 1, len(utds))
        colors = [plt.cm.viridis(i) for i in indices]
    else:
        colors = [plt.cm.viridis(0.5)]  # Use middle color if only one UTD
    fig, axes = plt.subplots(
        len(batch_sizes), len(envs), figsize=(len(envs) * 4, len(batch_sizes) * 3)
    )
    axes = np.array(axes).reshape(len(batch_sizes), len(envs))

    lines, labels = [], []

    for i, batch_size in enumerate(batch_sizes):
        for j, env in enumerate(envs):
            ax = axes[i, j]
            if filter_bs is not None:
                ax.set_title(env)
            else:
                ax.set_title(f'{env}, bs={batch_size}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Critic Width')
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xticks(critic_widths)
            ax.set_xticklabels(critic_widths)

            slope_lines, slope_labels = [], []

            for utd, color in zip(utds, colors):
                subset = df[
                    (df['env_name'] == env) & (df['batch_size'] == batch_size) & (df['utd'] == utd)
                ]
                if len(subset) == 0:
                    continue
                label = f'$\sigma$={utd}'
                (line,) = ax.plot(
                    subset['critic_width'],
                    subset['summary'],
                    'o',
                    label=label,
                    color=color,
                    alpha=0.3,  # More translucent points
                )
                linear_fit = sm.OLS(
                    np.log(subset['summary']),
                    sm.add_constant(np.log(subset['critic_width'])),
                ).fit()
                smooth_x = np.linspace(
                    np.log(subset['critic_width'].min()),
                    np.log(subset['critic_width'].max()),
                    100,
                )
                smooth_y = linear_fit.predict(sm.add_constant(smooth_x))
                (line,) = ax.plot(
                    np.exp(smooth_x),
                    np.exp(smooth_y),
                    color=color,
                    linestyle='--',
                    alpha=1,
                    linewidth=2.5,  # Thicker lines
                )

                # print(linear_fit.params)
                # slope = linear_fit.params.iloc[1]
                # r2 = linear_fit.rsquared
                # slope_lines.append(line)
                # slope_labels.append(f'slope={slope:.2f}, R²={r2:.2f}')

                if label not in labels:
                    labels.append(label)
                    lines.append(line)

            ax.legend(slope_lines, slope_labels, fontsize=8, frameon=False)

    sorted_idx = sorted(range(len(labels)), key=lambda i: float(labels[i].split('=')[1]))
    lines = [lines[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(labels))
    if filter_bs is None:
        fig.suptitle(f'{metric}, {summarize_how}', fontsize=16)
    else:
        fig.suptitle(f'{metric}, bs={filter_bs}, {summarize_how}', fontsize=16)
    plt.tight_layout()


def _solve_sigma_n_given_d_c_loss(
    utd_and_critic_params,
    data,
    compute,
    a,
    alpha,
    b,
    beta,
    c,
    utd_m,
    utd_s,
    critic_params_m,
    critic_params_s,
):
    batch_size = 256  # TODO
    u, v = utd_and_critic_params
    utd = _log_rescale_inverse(softplus(u), utd_m, utd_s)
    critic_params = _log_rescale_inverse(softplus(v), critic_params_m, critic_params_s)
    fit_data = sum_of_powerlaw_fn(utd, critic_params, a, alpha, b, beta, c)
    fit_compute = 10 * batch_size * utd * critic_params * fit_data
    data_loss = np.abs(np.log(data) - np.log(fit_data))
    compute_loss = np.abs(np.log(compute) - np.log(fit_compute))
    return 5 * data_loss + compute_loss


def solve_sigma_n_given_d_c(
    sample_utds, sample_critic_params, data: float, compute: float, a, alpha, b, beta, c
):
    _, utd_m, utd_s = _log_rescale(sample_utds)
    _, critic_params_m, critic_params_s = _log_rescale(sample_critic_params)
    args = (data, compute, a, alpha, b, beta, c, utd_m, utd_s, critic_params_m, critic_params_s)
    init_grid = [slice(0.0, 0.0, 1.0), slice(0.0, 0.0, 1.0)]
    u, v = _generic_fit_scipy(_solve_sigma_n_given_d_c_loss, args, init_grid, precise=True)
    utd = _log_rescale_inverse(softplus(u), utd_m, utd_s)
    critic_params = _log_rescale_inverse(softplus(v), critic_params_m, critic_params_s)
    return utd, critic_params


def insert_budget_fitted_hparams(multiple_budget_optimal_hparams, params_per_thresh):
    batch_size = 256
    results = {}
    for attr in ['data_efficiency', 'compute']:
        results[f'budget_to_{attr}'] = {}
        for env in multiple_budget_optimal_hparams['env_name'].unique():
            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            x = env_data['opt_budget']
            y = env_data[attr]
            slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
            line = np.exp(intercept) * x**slope
            r2 = r_squared(np.log(y), np.log(line))
            results[f'budget_to_{attr}'][env] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r2,
            }

    for i, row in multiple_budget_optimal_hparams.iterrows():
        env = row['env_name']
        data_from_budget = np.exp(
            results['budget_to_data_efficiency'][env]['slope'] * np.log(row['opt_budget'])
            + results['budget_to_data_efficiency'][env]['intercept']
        )
        compute_from_budget = np.exp(
            results['budget_to_compute'][env]['slope'] * np.log(row['opt_budget'])
            + results['budget_to_compute'][env]['intercept']
        )
        a, alpha, b, beta, c = params_per_thresh[row['threshold_idx']][env][:5]
        utd, critic_params = solve_sigma_n_given_d_c(
            [row['opt_utd']],
            [row['opt_critic_params']],
            data_from_budget,
            compute_from_budget,
            a,
            alpha,
            b,
            beta,
            c,
        )
        multiple_budget_optimal_hparams.loc[i, 'budget_fitted_utd'] = utd
        multiple_budget_optimal_hparams.loc[i, 'budget_fitted_critic_params'] = critic_params
        multiple_budget_optimal_hparams.loc[i, 'budget_fitted_data_efficiency'] = data_from_budget
        multiple_budget_optimal_hparams.loc[i, 'budget_fitted_compute'] = compute_from_budget
        multiple_budget_optimal_hparams.loc[i, 'budget_fitted_utd_n_data'] = sum_of_powerlaw_fn(
            utd, critic_params, a, alpha, b, beta, c
        )
        multiple_budget_optimal_hparams.loc[i, 'budget_fitted_utd_n_compute'] = (
            10
            * batch_size
            * utd
            * critic_params
            * multiple_budget_optimal_hparams.loc[i, 'budget_fitted_data_efficiency']
        )

    return multiple_budget_optimal_hparams


def plot_multiple_budget_optimal(
    multiple_budget_optimal_hparams, delta_dict, thresholds_per_env, loss_type='l2'
):
    multiple_budget_optimal_hparams['threshold'] = multiple_budget_optimal_hparams.apply(
        lambda row: thresholds_per_env[row['env_name']][row['threshold_idx']], axis=1
    )
    envs = sorted(multiple_budget_optimal_hparams['env_name'].unique())
    n_cols = min(4, len(envs))
    n_rows = int(np.ceil(len(envs) / n_cols))
    delta_latex = r'$\delta$'

    # Plot 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for env, ax in zip(envs, axes):
        env_data = multiple_budget_optimal_hparams[
            multiple_budget_optimal_hparams['env_name'] == env
        ]
        ax.scatter(
            env_data['opt_utd'],
            env_data['opt_critic_params'],
            c=env_data['threshold'],
            cmap='viridis',
        )
        # if 'budget_fitted_utd' in env_data.columns:
        if False:
            points = np.array(
                [env_data['budget_fitted_utd'], env_data['budget_fitted_critic_params']]
            ).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
            lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.5)
            lc.set_array(env_data['threshold'])
            ax.add_collection(lc)
        ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})')
        ax.set_xlabel('UTD')
        ax.set_ylabel('Critic params')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        cmap = plt.cm.viridis
        norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('Threshold', rotation=-90, va='bottom')

    plt.suptitle(r'$(C+\delta D)$-optimal UTD and model size per threshold', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot 2
    for attr, title in [
        ('data_efficiency', r'$D$ at optimal $C+\delta D$'),
        ('compute', r'$C$ at optimal $C+\delta D$'),
    ]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
        axes = np.array(axes).reshape(-1)

        for env, ax in zip(envs, axes):
            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            x = env_data['opt_budget']
            y = env_data[attr]
            if loss_type == 'l2':
                slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
            elif loss_type == 'l1':
                slope, intercept = curve_fit(
                    lambda t, a, b: a * t + b, np.log(x), np.log(y), method='trf', loss='soft_l1'
                )[0]
            else:
                raise ValueError(f'Invalid loss type: {loss_type}')
            line = np.exp(intercept) * x**slope
            r2 = r_squared(np.log(y), np.log(line))
            ax.scatter(x, y, c=env_data['threshold'], cmap='viridis')
            ax.plot(x, line, color='gray', linestyle='--', label=f'R² = {r2:.2f}')
            ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Budget')
            ax.set_ylabel(attr)
            ax.legend(frameon=False, loc='upper left')
            cmap = plt.cm.viridis
            norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax)
            cbar.ax.set_ylabel('Threshold', rotation=-90, va='bottom')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot 2
    for attr, title in [
        ('opt_utd', r'$\sigma$ at optimal $C+\delta D$'),
        ('opt_critic_params', r'$N$ at optimal $C+\delta D$'),
    ]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
        axes = np.array(axes).reshape(-1)

        for env, ax in zip(envs, axes):
            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            x = env_data['opt_budget']
            y = env_data[attr]

            slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
            line = np.exp(intercept) * x**slope
            r2 = r_squared(np.log(y), np.log(line))
            ax.scatter(x, y, c=env_data['threshold'], cmap='viridis')
            ax.plot(x, line, color='gray', linestyle='--', label=f'R² = {r2:.2f}')
            ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Budget')
            ax.set_ylabel(attr)
            ax.legend(frameon=False, loc='upper left')
            cmap = plt.cm.viridis
            norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax)
            cbar.ax.set_ylabel('Threshold', rotation=-90, va='bottom')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot 3
    # for attr, title in [
    #     ('opt_budget', r'Optimal $C+\delta D$ per threshold'),
    #     ('data_efficiency', r'$D$ at optimal $C+\delta D$ per threshold'),
    #     ('compute', r'$C$ at optimal $C+\delta D$ per threshold'),
    # ]:
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    #     axes = np.array(axes).reshape(-1)

    #     for env, ax in zip(envs, axes):
    #         env_data = multiple_budget_optimal_hparams[multiple_budget_optimal_hparams['env_name'] == env]
    #         x = env_data['threshold']
    #         y = env_data[attr]
    #         slope, intercept = np.polyfit(x, y, 1)
    #         line = slope * x + intercept

    #         y_mean = np.mean(y)
    #         ss_total = np.sum((y - y_mean) ** 2)
    #         ss_residual = np.sum((y - line) ** 2)
    #         r_squared = 1 - (ss_residual / ss_total)

    #         ax.scatter(x, y)
    #         ax.plot(x, line, color='red', label=f'slope = {slope:.2e}, R² = {r_squared:.2f}')
    #         ax.legend()
    #         ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})')
    #         ax.set_xlabel('Threshold')
    #     plt.suptitle(title, fontsize=14)
    #     plt.tight_layout()
    #     plt.show()

    for attr, title_prefix in [
        ('opt_utd', 'UTD'),
        ('opt_critic_params', 'Critic params'),
    ]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
        axes = np.array(axes).reshape(-1)

        for env, ax in zip(envs, axes):
            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            x = env_data['threshold']
            y = env_data[attr]
            ax.scatter(x, y)
            ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})')
        plt.suptitle(title_prefix + r' at optimal $C + \delta D$ per threshold', fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_budget_data_compute_opt_pretty(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    data_yticks=None,
    data_yscale=None,
    compute_yticks=None,
    compute_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
    cd_save_path=None,
):
    multiple_budget_optimal_hparams['threshold'] = multiple_budget_optimal_hparams.apply(
        lambda row: thresholds_per_env[row['env_name']][row['threshold_idx']], axis=1
    )
    envs = sorted(multiple_budget_optimal_hparams['env_name'].unique())
    n_cols = min(4, len(envs))
    n_rows = int(np.ceil(len(envs) / n_cols))
    assert n_rows == 1
    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [qscaled_plot_utils.COLORS[0], qscaled_plot_utils.COLORS[1]]
    )
    delta_latex = r'$\delta$'

    # Plot 1: x = budget, y = data or compute

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 4, n_rows * 3.5 * 2))

    for i, (attr, label) in enumerate(
        [
            (
                'data_efficiency',
                r'$\mathcal{D}_{\mathcal{F}_J^*}$: Optimal data',
            ),
            (
                'compute',
                r'$\mathcal{C}_{\mathcal{F}_J^*}$: Optimal compute',
            ),
        ]
    ):
        axes_ = axes[i]

        for j, (env, ax) in enumerate(zip(envs, axes_)):
            if isinstance(xticks, list) or xticks is None:
                env_xticks = xticks
            elif isinstance(xticks, dict):
                env_xticks = xticks[env]
            if isinstance(xscale, str) or xscale is None:
                env_xscale = xscale
            elif isinstance(xscale, dict):
                env_xscale = xscale[env]
            if isinstance(data_yticks, list) or data_yticks is None:
                env_data_yticks = data_yticks
            elif isinstance(data_yticks, dict):
                env_data_yticks = data_yticks[env]
            if isinstance(data_yscale, str) or data_yscale is None:
                env_data_yscale = data_yscale
            elif isinstance(data_yscale, dict):
                env_data_yscale = data_yscale[env]
            if isinstance(compute_yticks, list) or compute_yticks is None:
                env_compute_yticks = compute_yticks
            elif isinstance(compute_yticks, dict):
                env_compute_yticks = compute_yticks[env]
            if isinstance(compute_yscale, str) or compute_yscale is None:
                env_compute_yscale = compute_yscale
            elif isinstance(compute_yscale, dict):
                env_compute_yscale = compute_yscale[env]

            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            env_data = env_data.sort_values(by='opt_budget')
            x = env_data['opt_budget']
            y = env_data[attr]
            thresholds = env_data['threshold']
            norm = plt.Normalize(thresholds.min(), thresholds.max())
            x_fit = x[:-num_extrapolated_points]
            y_fit = y[:-num_extrapolated_points]
            thresholds_fit = thresholds[:-num_extrapolated_points]
            colors_fit = cmap(norm(thresholds_fit))
            x_extrapolated = x[-num_extrapolated_points:]
            y_extrapolated = y[-num_extrapolated_points:]
            thresholds_extrapolated = thresholds[-num_extrapolated_points:]
            colors_extrapolated = cmap(norm(thresholds_extrapolated))
            slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
            line = np.exp(intercept) * x**slope
            r2 = r_squared(np.log(y), np.log(line))
            ax.scatter(x_fit, y_fit, c=colors_fit, s=100)
            ax.scatter(x_extrapolated, y_extrapolated, c=colors_extrapolated, s=100, marker='X')
            ax.plot(
                x, line, color='gray', linestyle='--', label=r'$R^2$' + f'={r2:.2f}', linewidth=3
            )

            # Calculate confidence intervals
            y_fit_pred = np.exp(intercept) * x_fit**slope
            residuals = np.log(y_fit) - np.log(y_fit_pred)
            mean_x = np.mean(np.log(x_fit))
            n = len(x_fit)
            t_value = 1.96  # for 95% confidence interval
            s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
            conf_interval = (
                t_value
                * s_err
                * np.sqrt(1 / n + (np.log(x) - mean_x) ** 2 / np.sum((np.log(x_fit) - mean_x) ** 2))
            )
            lower_bound = np.exp(np.log(line) - conf_interval)
            upper_bound = np.exp(np.log(line) + conf_interval)

            # Plot confidence interval as a light gray band
            ax.fill_between(x, lower_bound, upper_bound, color='lightgray', alpha=0.5, zorder=-1)

            if i == 0:
                # ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})', fontsize='xx-large', y=1.05)
                ax.set_title(f'{env}', fontsize='xx-large', y=1.05)
            ax.set_xscale('log')
            ax.set_yscale('log')
            norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # cbar.ax.set_ylabel('Threshold', rotation=-90, va="bottom")

            ax.legend(prop={'size': 14}, ncol=1, frameon=False, loc='lower right')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            cbar = plt.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize='xx-large')
            if j == len(envs) - 1:
                cbar.set_label('$J$: Performance level', size='xx-large')

            rliable_plot_utils._annotate_and_decorate_axis(
                ax,
                xlabel=r'$\mathcal{F}_J^*$: Optimal budget for $J$' if i == 1 else '',
                ylabel=label if j == 0 else '',
                labelsize='xx-large',
                ticklabelsize='xx-large',
                grid_alpha=0.2,
                legend=False,
            )
            ax.yaxis.set_label_coords(-0.25, 0.5)

            qscaled_plot_utils.ax_set_x_bounds_and_scale(ax, xticks=env_xticks, xscale=env_xscale)
            if attr == 'data_efficiency':
                qscaled_plot_utils.ax_set_y_bounds_and_scale(
                    ax, yticks=env_data_yticks, yscale=env_data_yscale
                )
            elif attr == 'compute':
                qscaled_plot_utils.ax_set_y_bounds_and_scale(
                    ax, yticks=env_compute_yticks, yscale=env_compute_yscale
                )

        # plt.suptitle(title, fontsize=14)
    plt.tight_layout(h_pad=4.0)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

    # Plot 2: x = data, y = compute

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    xlabel = r'$\mathcal{D}_{\mathcal{F}_J^*}$: Optimal data'
    ylabel = r'$\mathcal{C}_{\mathcal{F}_J^*}$: Optimal compute'
    for j, (env, ax) in enumerate(zip(envs, axes)):
        if isinstance(data_yticks, list) or data_yticks is None:
            env_data_ticks = data_yticks
        elif isinstance(data_yticks, dict):
            env_data_ticks = data_yticks[env]
        if isinstance(data_yscale, str) or data_yscale is None:
            env_data_scale = data_yscale
        elif isinstance(data_yscale, dict):
            env_data_scale = data_yscale[env]
        if isinstance(compute_yticks, list) or compute_yticks is None:
            env_compute_ticks = compute_yticks
        elif isinstance(compute_yticks, dict):
            env_compute_ticks = compute_yticks[env]
        if isinstance(compute_yscale, str) or compute_yscale is None:
            env_compute_scale = compute_yscale
        elif isinstance(compute_yscale, dict):
            env_compute_scale = compute_yscale[env]

        env_data = multiple_budget_optimal_hparams[
            multiple_budget_optimal_hparams['env_name'] == env
        ]
        env_data = env_data.sort_values(by='opt_budget')
        x = env_data['data_efficiency']
        y = env_data['compute']
        thresholds = env_data['threshold']
        norm = plt.Normalize(thresholds.min(), thresholds.max())
        x_fit = x[:-num_extrapolated_points]
        y_fit = y[:-num_extrapolated_points]
        thresholds_fit = thresholds[:-num_extrapolated_points]
        colors_fit = cmap(norm(thresholds_fit))
        x_extrapolated = x[-num_extrapolated_points:]
        y_extrapolated = y[-num_extrapolated_points:]
        thresholds_extrapolated = thresholds[-num_extrapolated_points:]
        colors_extrapolated = cmap(norm(thresholds_extrapolated))
        slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
        line = np.exp(intercept) * x**slope
        r2 = r_squared(np.log(y), np.log(line))
        ax.scatter(x_fit, y_fit, c=colors_fit, s=100)
        ax.scatter(x_extrapolated, y_extrapolated, c=colors_extrapolated, s=100, marker='X')
        ax.plot(x, line, color='gray', linestyle='--', label=r'$R^2$' + f'={r2:.2f}', linewidth=3)

        # Calculate confidence intervals
        y_fit_pred = np.exp(intercept) * x_fit**slope
        residuals = np.log(y_fit) - np.log(y_fit_pred)
        mean_x = np.mean(np.log(x_fit))
        n = len(x_fit)
        t_value = 1.96  # for 95% confidence interval
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
        conf_interval = (
            t_value
            * s_err
            * np.sqrt(1 / n + (np.log(x) - mean_x) ** 2 / np.sum((np.log(x_fit) - mean_x) ** 2))
        )
        lower_bound = np.exp(np.log(line) - conf_interval)
        upper_bound = np.exp(np.log(line) + conf_interval)

        # Plot confidence interval as a light gray band
        ax.fill_between(x, lower_bound, upper_bound, color='lightgray', alpha=0.5, zorder=-1)

        if i == 0:
            # ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})', fontsize='xx-large', y=1.05)
            ax.set_title(f'{env}', fontsize='xx-large', y=1.05)
        ax.set_xscale('log')
        ax.set_yscale('log')
        norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar.ax.set_ylabel('Threshold', rotation=-90, va="bottom")

        ax.legend(prop={'size': 14}, ncol=1, frameon=False, loc='lower right')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.2)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize='xx-large')
        if j == len(envs) - 1:
            cbar.set_label('$J$: Performance level', size='xx-large')

        rliable_plot_utils._annotate_and_decorate_axis(
            ax,
            xlabel=xlabel,
            ylabel=ylabel if j == 0 else '',
            labelsize='xx-large',
            ticklabelsize='xx-large',
            grid_alpha=0.2,
            legend=False,
        )
        ax.yaxis.set_label_coords(-0.25, 0.5)

        qscaled_plot_utils.ax_set_x_bounds_and_scale(
            ax, xticks=env_data_ticks, xscale=env_data_scale
        )
        qscaled_plot_utils.ax_set_y_bounds_and_scale(
            ax, yticks=env_compute_ticks, yscale=env_compute_scale
        )

        # plt.suptitle(title, fontsize=14)
    plt.tight_layout(h_pad=4.0)
    if cd_save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(cd_save_path)
    plt.show()


def _plot_budget_data_or_compute_opt_pretty_helper(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    include_data=True,
    include_compute=True,
    data_yticks=None,
    data_yscale=None,
    compute_yticks=None,
    compute_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    assert include_data or include_compute
    assert not (include_data and include_compute)

    original_rc_params = plt.rcParams.copy()
    qscaled_plot_utils.set_theme()

    multiple_budget_optimal_hparams['threshold'] = multiple_budget_optimal_hparams.apply(
        lambda row: thresholds_per_env[row['env_name']][row['threshold_idx']], axis=1
    )
    envs = sorted(multiple_budget_optimal_hparams['env_name'].unique())
    n_cols = min(4, len(envs))
    n_rows = int(np.ceil(len(envs) / n_cols))
    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [qscaled_plot_utils.COLORS[0], qscaled_plot_utils.COLORS[1]]
    )
    delta_latex = r'$\delta$'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = np.array(axes).reshape(-1)

    if include_data:
        attr = 'data_efficiency'
        label = r'$\mathcal{D}_{\mathcal{F}_J^*}$: Optimal data'
    else:
        attr = 'compute'
        label = r'$\mathcal{C}_{\mathcal{F}_J^*}$: Optimal compute'

    for i, (env, ax) in enumerate(zip(envs, axes)):
        j_ = i % n_cols
        i_ = i // n_cols
        if isinstance(xticks, list) or xticks is None:
            env_xticks = xticks
        elif isinstance(xticks, dict):
            env_xticks = xticks[env]
        if isinstance(xscale, str) or xscale is None:
            env_xscale = xscale
        elif isinstance(xscale, dict):
            env_xscale = xscale[env]
        if include_data:
            if isinstance(data_yticks, list) or data_yticks is None:
                env_data_yticks = data_yticks
            elif isinstance(data_yticks, dict):
                env_data_yticks = data_yticks[env]
            if isinstance(data_yscale, str) or data_yscale is None:
                env_data_yscale = data_yscale
            elif isinstance(data_yscale, dict):
                env_data_yscale = data_yscale[env]
        else:
            if isinstance(compute_yticks, list) or compute_yticks is None:
                env_data_yticks = compute_yticks
            elif isinstance(compute_yticks, dict):
                env_data_yticks = compute_yticks[env]
            if isinstance(compute_yscale, str) or compute_yscale is None:
                env_data_yscale = compute_yscale
            elif isinstance(compute_yscale, dict):
                env_data_yscale = compute_yscale[env]

        env_data = multiple_budget_optimal_hparams[
            multiple_budget_optimal_hparams['env_name'] == env
        ]
        env_data = env_data.sort_values(by='opt_budget')
        x = env_data['opt_budget']
        y = env_data[attr]
        thresholds = env_data['threshold']
        norm = plt.Normalize(thresholds.min(), thresholds.max())
        x_fit = x[:-num_extrapolated_points]
        y_fit = y[:-num_extrapolated_points]
        thresholds_fit = thresholds[:-num_extrapolated_points]
        colors_fit = cmap(norm(thresholds_fit))
        x_extrapolated = x[-num_extrapolated_points:]
        y_extrapolated = y[-num_extrapolated_points:]
        thresholds_extrapolated = thresholds[-num_extrapolated_points:]
        colors_extrapolated = cmap(norm(thresholds_extrapolated))
        slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
        line = np.exp(intercept) * x**slope
        r2 = r_squared(np.log(y), np.log(line))
        ax.scatter(x_fit, y_fit, c=colors_fit, s=100)
        ax.scatter(x_extrapolated, y_extrapolated, c=colors_extrapolated, s=100, marker='X')
        ax.plot(x, line, color='gray', linestyle='--', label=r'$R^2$' + f'={r2:.2f}', linewidth=3)

        # Calculate confidence intervals
        y_fit_pred = np.exp(intercept) * x_fit**slope
        residuals = np.log(y_fit) - np.log(y_fit_pred)
        mean_x = np.mean(np.log(x_fit))
        n = len(x_fit)
        t_value = 1.96  # for 95% confidence interval
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
        conf_interval = (
            t_value
            * s_err
            * np.sqrt(1 / n + (np.log(x) - mean_x) ** 2 / np.sum((np.log(x_fit) - mean_x) ** 2))
        )
        lower_bound = np.exp(np.log(line) - conf_interval)
        upper_bound = np.exp(np.log(line) + conf_interval)

        # Plot confidence interval as a light gray band
        ax.fill_between(x, lower_bound, upper_bound, color='lightgray', alpha=0.5, zorder=-1)

        # if i_ == 0:
        # ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})', fontsize='xx-large', y=1.05)
        ax.set_title(f'{env}', fontsize='xx-large', y=1.05)
        ax.set_xscale('log')
        ax.set_yscale('log')
        norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar.ax.set_ylabel('Threshold', rotation=-90, va="bottom")

        ax.legend(prop={'size': 14}, ncol=1, frameon=False, loc='lower right')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.2)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize='xx-large')
        if j_ == n_cols - 1:
            cbar.set_label('$J$: Performance level', size='xx-large')

        rliable_plot_utils._annotate_and_decorate_axis(
            ax,
            xlabel=r'$\mathcal{F}_J^*$: Optimal budget for $J$' if i_ == n_rows - 1 else '',
            # ylabel=label if j_ == 0 else '',
            labelsize='xx-large',
            ticklabelsize='xx-large',
            grid_alpha=0.2,
            legend=False,
        )
        ax.yaxis.set_label_coords(-0.35, 0.5)
        ax.set_ylabel(label if j_ == 0 else '', size='xx-large')

        qscaled_plot_utils.ax_set_x_bounds_and_scale(ax, xticks=env_xticks, xscale=env_xscale)
        qscaled_plot_utils.ax_set_y_bounds_and_scale(
            ax, yticks=env_data_yticks, yscale=env_data_yscale
        )

    for i in range(len(envs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

    plt.rcParams = original_rc_params


def plot_budget_data_opt_pretty(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    data_yticks=None,
    data_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    _plot_budget_data_or_compute_opt_pretty_helper(
        multiple_budget_optimal_hparams,
        delta_dict,
        thresholds_per_env,
        xticks,
        xscale,
        include_data=True,
        include_compute=False,
        data_yticks=data_yticks,
        data_yscale=data_yscale,
        num_extrapolated_points=num_extrapolated_points,
        save_path=save_path,
    )


def plot_budget_compute_opt_pretty(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    compute_yticks=None,
    compute_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    _plot_budget_data_or_compute_opt_pretty_helper(
        multiple_budget_optimal_hparams,
        delta_dict,
        thresholds_per_env,
        xticks,
        xscale,
        include_data=False,
        include_compute=True,
        compute_yticks=compute_yticks,
        compute_yscale=compute_yscale,
        num_extrapolated_points=num_extrapolated_points,
        save_path=save_path,
    )


def plot_budget_n_sigma_opt_pretty(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    sigma_yticks=None,
    sigma_yscale=None,
    n_yticks=None,
    n_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    original_rc_params = plt.rcParams.copy()
    qscaled_plot_utils.set_theme()

    multiple_budget_optimal_hparams['threshold'] = multiple_budget_optimal_hparams.apply(
        lambda row: thresholds_per_env[row['env_name']][row['threshold_idx']], axis=1
    )
    envs = sorted(multiple_budget_optimal_hparams['env_name'].unique())
    n_cols = min(4, len(envs))
    n_rows = int(np.ceil(len(envs) / n_cols))
    assert n_rows == 1
    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [qscaled_plot_utils.COLORS[0], qscaled_plot_utils.COLORS[1]]
    )
    delta_latex = r'$\delta$'

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 4.5, n_rows * 3.75 * 2))

    for i, (attr, label) in enumerate(
        [
            (
                'opt_utd',
                r'$\sigma_{\mathcal{F}_J}$: Optimal UTD',
            ),
            (
                'opt_critic_params',
                r'$N_{\mathcal{F}_J}$: Optimal model size',
            ),
        ]
    ):
        axes_ = axes[i]

        for j, (env, ax) in enumerate(zip(envs, axes_)):
            if isinstance(xticks, list) or xticks is None:
                env_xticks = xticks
            elif isinstance(xticks, dict):
                env_xticks = xticks[env]
            if isinstance(xscale, str) or xscale is None:
                env_xscale = xscale
            elif isinstance(xscale, dict):
                env_xscale = xscale[env]
            if isinstance(sigma_yticks, list) or sigma_yticks is None:
                env_sigma_yticks = sigma_yticks
            elif isinstance(sigma_yticks, dict):
                env_sigma_yticks = sigma_yticks[env]
            if isinstance(sigma_yscale, str) or sigma_yscale is None:
                env_sigma_yscale = sigma_yscale
            elif isinstance(sigma_yscale, dict):
                env_sigma_yscale = sigma_yscale[env]
            if isinstance(n_yticks, list) or n_yticks is None:
                env_n_yticks = n_yticks
            elif isinstance(n_yticks, dict):
                env_n_yticks = n_yticks[env]
            if isinstance(n_yscale, str) or n_yscale is None:
                env_n_yscale = n_yscale
            elif isinstance(n_yscale, dict):
                env_n_yscale = n_yscale[env]

            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            x = env_data['opt_budget']
            y = env_data[attr]
            thresholds = env_data['threshold']
            norm = plt.Normalize(thresholds.min(), thresholds.max())
            x_fit = x[:-num_extrapolated_points]
            y_fit = y[:-num_extrapolated_points]
            thresholds_fit = thresholds[:-num_extrapolated_points]
            colors_fit = cmap(norm(thresholds_fit))
            x_extrapolated = x[-num_extrapolated_points:]
            y_extrapolated = y[-num_extrapolated_points:]
            thresholds_extrapolated = thresholds[-num_extrapolated_points:]
            colors_extrapolated = cmap(norm(thresholds_extrapolated))
            slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
            line = np.exp(intercept) * x**slope
            r2 = r_squared(np.log(y), np.log(line))
            # Calculate confidence intervals
            y_fit_pred = np.exp(intercept) * x**slope
            residuals = np.log(y) - np.log(y_fit_pred)
            mean_x = np.mean(np.log(x))
            n = len(x)
            t_value = 1.96  # for 95% confidence interval
            s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
            conf_interval = (
                t_value
                * s_err
                * np.sqrt(1 / n + (np.log(x) - mean_x) ** 2 / np.sum((np.log(x) - mean_x) ** 2))
            )
            lower_bound = np.exp(np.log(line) - conf_interval)
            upper_bound = np.exp(np.log(line) + conf_interval)

            # Plot confidence interval as a light gray band
            ax.fill_between(x, lower_bound, upper_bound, color='lightgray', alpha=0.5, zorder=-1)

            ax.scatter(x_fit, y_fit, c=colors_fit, s=100)
            ax.scatter(x_extrapolated, y_extrapolated, c=colors_extrapolated, s=100, marker='X')
            ax.plot(
                x, line, color='gray', linestyle='--', label=r'$R^2$' + f'={r2:.2f}', linewidth=3
            )
            if i == 0:
                # ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})', fontsize='xx-large', y=1.05)
                ax.set_title(f'{env}', fontsize='xx-large', y=1.05)
            ax.set_xscale('log')
            ax.set_yscale('log')
            norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # cbar.ax.set_ylabel('Threshold', rotation=-90, va="bottom")

            ax.legend(prop={'size': 14}, ncol=1, frameon=False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            cbar = plt.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize='xx-large')
            if j == len(envs) - 1:
                cbar.set_label('$J$: Performance level', size='xx-large')

            rliable_plot_utils._annotate_and_decorate_axis(
                ax,
                xlabel=r'$\mathcal{F}_J^*$: Optimal budget for $J$' if i == 1 else '',
                ylabel=label if j == 0 else '',
                labelsize='xx-large',
                ticklabelsize='xx-large',
                grid_alpha=0.2,
                legend=False,
            )
            ax.yaxis.set_label_coords(-0.25, 0.5)

            qscaled_plot_utils.ax_set_x_bounds_and_scale(ax, xticks=env_xticks, xscale=env_xscale)
            if attr == 'opt_utd':
                qscaled_plot_utils.ax_set_y_bounds_and_scale(
                    ax, yticks=env_sigma_yticks, yscale=env_sigma_yscale, yfloat=True
                )
            elif attr == 'opt_critic_params':
                qscaled_plot_utils.ax_set_y_bounds_and_scale(
                    ax, yticks=env_n_yticks, yscale=env_n_yscale
                )

        # plt.suptitle(title, fontsize=14)
    plt.tight_layout(h_pad=4.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

    plt.rcParams = original_rc_params


def _plot_budget_n_or_sigma_opt_pretty_helper(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    include_sigma=True,
    include_n=True,
    sigma_yticks=None,
    sigma_yscale=None,
    n_yticks=None,
    n_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    assert include_sigma or include_n
    assert not (include_sigma and include_n)

    original_rc_params = plt.rcParams.copy()
    qscaled_plot_utils.set_theme()

    multiple_budget_optimal_hparams['threshold'] = multiple_budget_optimal_hparams.apply(
        lambda row: thresholds_per_env[row['env_name']][row['threshold_idx']], axis=1
    )
    envs = sorted(multiple_budget_optimal_hparams['env_name'].unique())
    n_cols = min(4, len(envs))
    n_rows = int(np.ceil(len(envs) / n_cols))
    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [qscaled_plot_utils.COLORS[0], qscaled_plot_utils.COLORS[1]]
    )
    delta_latex = r'$\delta$'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.75))
    axes = np.array(axes).reshape(-1)

    if include_sigma:
        attr = 'opt_utd'
        label = r'$\sigma_{\mathcal{F}_J}$: Optimal UTD'
    else:
        attr = 'opt_critic_params'
        label = r'$N_{\mathcal{F}_J}$: Optimal model size'

    for i, (env, ax) in enumerate(zip(envs, axes)):
        i_ = i // n_cols
        j_ = i % n_cols
        if isinstance(xticks, list) or xticks is None:
            env_xticks = xticks
        elif isinstance(xticks, dict):
            env_xticks = xticks[env]
        if isinstance(xscale, str) or xscale is None:
            env_xscale = xscale
        elif isinstance(xscale, dict):
            env_xscale = xscale[env]
        if isinstance(sigma_yticks, list) or sigma_yticks is None:
            env_sigma_yticks = sigma_yticks
        elif isinstance(sigma_yticks, dict):
            env_sigma_yticks = sigma_yticks[env]
        if isinstance(sigma_yscale, str) or sigma_yscale is None:
            env_sigma_yscale = sigma_yscale
        elif isinstance(sigma_yscale, dict):
            env_sigma_yscale = sigma_yscale[env]
        if isinstance(n_yticks, list) or n_yticks is None:
            env_n_yticks = n_yticks
        elif isinstance(n_yticks, dict):
            env_n_yticks = n_yticks[env]
        if isinstance(n_yscale, str) or n_yscale is None:
            env_n_yscale = n_yscale
        elif isinstance(n_yscale, dict):
            env_n_yscale = n_yscale[env]

        env_data = multiple_budget_optimal_hparams[
            multiple_budget_optimal_hparams['env_name'] == env
        ]
        x = env_data['opt_budget']
        y = env_data[attr]
        thresholds = env_data['threshold']
        norm = plt.Normalize(thresholds.min(), thresholds.max())
        x_fit = x[:-num_extrapolated_points]
        y_fit = y[:-num_extrapolated_points]
        thresholds_fit = thresholds[:-num_extrapolated_points]
        colors_fit = cmap(norm(thresholds_fit))
        x_extrapolated = x[-num_extrapolated_points:]
        y_extrapolated = y[-num_extrapolated_points:]
        thresholds_extrapolated = thresholds[-num_extrapolated_points:]
        colors_extrapolated = cmap(norm(thresholds_extrapolated))
        slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
        line = np.exp(intercept) * x**slope
        r2 = r_squared(np.log(y), np.log(line))
        # Calculate confidence intervals
        y_fit_pred = np.exp(intercept) * x**slope
        residuals = np.log(y) - np.log(y_fit_pred)
        mean_x = np.mean(np.log(x))
        n = len(x)
        t_value = 1.96  # for 95% confidence interval
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
        conf_interval = (
            t_value
            * s_err
            * np.sqrt(1 / n + (np.log(x) - mean_x) ** 2 / np.sum((np.log(x) - mean_x) ** 2))
        )
        lower_bound = np.exp(np.log(line) - conf_interval)
        upper_bound = np.exp(np.log(line) + conf_interval)

        # Plot confidence interval as a light gray band
        ax.fill_between(x, lower_bound, upper_bound, color='lightgray', alpha=0.5, zorder=-1)

        ax.scatter(x_fit, y_fit, c=colors_fit, s=100)
        ax.scatter(x_extrapolated, y_extrapolated, c=colors_extrapolated, s=100, marker='X')
        ax.plot(x, line, color='gray', linestyle='--', label=r'$R^2$' + f'={r2:.2f}', linewidth=3)
        # if i_ == 0:
        # ax.set_title(f'{env} ({delta_latex}={delta_dict[env]})', fontsize='xx-large', y=1.05)
        ax.set_title(f'{env}', fontsize='xx-large', y=1.05)
        ax.set_xscale('log')
        ax.set_yscale('log')
        norm = plt.Normalize(env_data['threshold'].min(), env_data['threshold'].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar.ax.set_ylabel('Threshold', rotation=-90, va="bottom")

        ax.legend(prop={'size': 14}, ncol=1, frameon=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.2)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize='xx-large')
        if j_ == n_cols - 1:
            cbar.set_label('$J$: Performance level', size='xx-large')

        rliable_plot_utils._annotate_and_decorate_axis(
            ax,
            xlabel=r'$\mathcal{F}_J^*$: Optimal budget for $J$' if i_ == n_rows - 1 else '',
            ylabel=label if j_ == 0 else '',
            labelsize='xx-large',
            ticklabelsize='xx-large',
            grid_alpha=0.2,
            legend=False,
        )
        ax.yaxis.set_label_coords(-0.25, 0.5)

        qscaled_plot_utils.ax_set_x_bounds_and_scale(ax, xticks=env_xticks, xscale=env_xscale)
        if attr == 'opt_utd':
            qscaled_plot_utils.ax_set_y_bounds_and_scale(
                ax, yticks=env_sigma_yticks, yscale=env_sigma_yscale, yfloat=True
            )
        elif attr == 'opt_critic_params':
            qscaled_plot_utils.ax_set_y_bounds_and_scale(
                ax, yticks=env_n_yticks, yscale=env_n_yscale
            )

    for i in range(len(envs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

    plt.rcParams = original_rc_params


def plot_budget_sigma_opt_pretty(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    sigma_yticks=None,
    sigma_yscale=None,
    n_yticks=None,
    n_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    _plot_budget_n_or_sigma_opt_pretty_helper(
        multiple_budget_optimal_hparams,
        delta_dict,
        thresholds_per_env,
        xticks,
        xscale,
        include_sigma=True,
        include_n=False,
        sigma_yticks=sigma_yticks,
        sigma_yscale=sigma_yscale,
        n_yticks=n_yticks,
        n_yscale=n_yscale,
        num_extrapolated_points=num_extrapolated_points,
        save_path=save_path,
    )


def plot_budget_n_opt_pretty(
    multiple_budget_optimal_hparams,
    delta_dict,
    thresholds_per_env,
    xticks=None,
    xscale=None,
    sigma_yticks=None,
    sigma_yscale=None,
    n_yticks=None,
    n_yscale=None,
    num_extrapolated_points=None,
    save_path=None,
):
    _plot_budget_n_or_sigma_opt_pretty_helper(
        multiple_budget_optimal_hparams,
        delta_dict,
        thresholds_per_env,
        xticks,
        xscale,
        include_sigma=False,
        include_n=True,
        sigma_yticks=sigma_yticks,
        sigma_yscale=sigma_yscale,
        n_yticks=n_yticks,
        n_yscale=n_yscale,
        num_extrapolated_points=num_extrapolated_points,
        save_path=save_path,
    )


def compute_budget_data_compute_equation(multiple_budget_optimal_hparams):
    envs = sorted(multiple_budget_optimal_hparams['env_name'].unique())
    results = []
    for attr in ['data_efficiency', 'compute']:
        result = {}
        for env in envs:
            env_data = multiple_budget_optimal_hparams[
                multiple_budget_optimal_hparams['env_name'] == env
            ]
            x = env_data['opt_budget']
            y = env_data[attr]
            slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
            line = 10 ** (intercept) * x**slope
            r2 = r_squared(np.log(y), np.log(line))
            result[env] = {'slope': slope, 'intercept': intercept, 'r_squared': r2}
        results.append(result)
    return results


def plot_multi_threshold_compute_optimal_hparams(df, n_thresholds):
    envs = sorted(df['env_name'].unique())
    n_cols = min(4, len(envs))
    n_rows = int(np.ceil(len(envs) / n_cols))

    # Plot 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    plt.suptitle('Grouped by threshold index', y=1.05)
    axes = np.array(axes).reshape(-1)
    threshold_idxs = range(n_thresholds)
    colors = sns.color_palette('viridis', n_colors=len(threshold_idxs))
    lines, labels = [], []
    for env, ax in zip(envs, axes):
        env_data = df[df['env_name'] == env]
        for threshold_idx, color in zip(threshold_idxs, colors):
            subset = env_data[env_data['threshold_idx'] == threshold_idx]
            label = f'Threshold {threshold_idx}'
            (line,) = ax.plot(
                subset['utd'], subset['critic_params'], marker='o', c=color, label=label
            )
            if label not in labels:
                lines.append(line)
                labels.append(label)
        ax.set_title(env)
        ax.set_xlabel('UTD')
        ax.set_ylabel('Critic params')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

    sorted_idx = [i for i, _ in sorted(enumerate(labels), key=lambda x: int(x[1].split(' ')[1]))]
    lines = [lines[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]
    fig.legend(
        lines,
        labels,
        loc='upper center',
        ncol=int(np.ceil(len(labels) / 2)),
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()

    # Plot 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    plt.suptitle('Grouped by compute budget', y=1.02)
    axes = np.array(axes).reshape(-1)
    compute_budgets = sorted(df['compute_budget'].unique())
    compute_budgets = [0.5e15, 16e15, 128e15, 1024e15]
    colors = sns.color_palette('viridis', n_colors=len(compute_budgets))
    lines, labels = [], []
    for env, ax in zip(envs, axes):
        env_data = df[df['env_name'] == env]
        for budget, color in zip(compute_budgets, colors):
            subset = env_data[env_data['compute_budget'] == budget].sort_values(by='threshold_idx')
            label = f'Budget {budget:.1e}'
            (line,) = ax.plot(
                subset['utd'], subset['critic_params'], marker='o', c=color, label=label
            )
            if label not in labels:
                lines.append(line)
                labels.append(label)
        ax.set_title(env)
        ax.set_xlabel('UTD')
        ax.set_ylabel('Critic params')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

    sorted_idx = [i for i, _ in sorted(enumerate(labels), key=lambda x: float(x[1].split(' ')[1]))]
    lines = [lines[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]
    fig.legend(
        lines,
        labels,
        loc='upper center',
        ncol=int(np.ceil(len(labels) / 2)),
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()
