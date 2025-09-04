import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jaxopt import LBFGS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from matplotlib.lines import Line2D
from copy import deepcopy

from scripts.utils import expand_log_range


def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def fit_quadratic(xs, ys):
    """y = a x^2 + b x + c"""
    a, b, c = np.polyfit(xs, ys, deg=2)
    return a, b, c


def inv_quadratic(y, a, b, c):
    """Returns the larger root"""
    return (-b + np.sqrt(b**2 - 4 * a * (c - y))) / (2 * a)


def params_to_width(fit_df, predict_df, predict_col):
    abcs = {}
    for env, group in fit_df.groupby('env_name'):
        abcs[env] = fit_quadratic(group['critic_width'], group['critic_params'])
    predictions = []
    for _, row in predict_df.iterrows():
        env = row['env_name']
        y = row[predict_col]
        a, b, c = abcs[env]
        predictions.append(inv_quadratic(y, a, b, c))
    return np.array(predictions)


def fit_regression(df, metric, interaction=True):
    """Actually also makes some plots"""
    df = deepcopy(df)
    df['end_of_training'] = df[metric].apply(lambda x: x[-1])
    envs = sorted(df['env_name'].unique())
    utds = sorted(df['utd'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    critic_widths = sorted(df['critic_width'].unique())
    colors = sns.color_palette('viridis', len(critic_widths))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', '+', 'x', '|', '_']
    color_map = {critic_width: colors[i] for i, critic_width in enumerate(critic_widths)}
    marker_map = {bs: markers[i] for i, bs in enumerate(batch_sizes)}

    """
    Linear fit
    """

    # linear_fit = []
    # fig, axes = plt.subplots(len(envs), len(utds), figsize=(3 * len(utds), 3 * len(envs)))

    # def linear_featurize(data):
    #     X = data[['critic_width', 'batch_size']].values
    #     X = np.column_stack([np.ones(len(X)), X, X[:, 0] * X[:, 1]])
    #     y = np.array(data['end_of_training'].values)
    #     return X, y

    # for i, env in enumerate(envs):
    #     for j, utd in enumerate(utds):
    #         subset = df.query(f'env_name=="{env}" and utd=={utd}')
    #         X, y = linear_featurize(subset)
    #         model = sm.OLS(y, X).fit()
    #         intercept, bs_coef, critic_width_coef, prod_coef = model.params
    #         predictions = model.predict(X)
    #         loss = np.mean((predictions - y) ** 2)
    #         r2 = model.rsquared

    #         linear_fit.append(
    #             {
    #                 'env': env,
    #                 'utd': utd,
    #                 'intercept': intercept,
    #                 'b_coef': bs_coef,
    #                 'w_coef': critic_width_coef,
    #                 'b*w_coef': prod_coef,
    #                 'loss': loss,
    #                 'formula': f'{metric} ~ {intercept:.3e} + {bs_coef:.3e} * B + {critic_width_coef:.3e} * width + {prod_coef:.3e} * B * width',
    #             }
    #         )

    #         ax = axes[i, j]
    #         graph_min, graph_max = float('inf'), float('-inf')
    #         for (critic_width, batch_size), group in subset.groupby(['critic_width', 'batch_size']):
    #             color = color_map[critic_width]
    #             marker = marker_map[batch_size]
    #             X, y = linear_featurize(group)
    #             predictions = model.predict(X)
    #             ax.scatter(
    #                 y,
    #                 predictions,
    #                 color=color,
    #                 marker=marker,
    #                 label=f'width={critic_width}, B={batch_size}',
    #             )
    #             graph_min = min(graph_min, y.min())
    #             graph_max = max(graph_max, y.max())
    #             graph_min = min(graph_min, predictions.min())
    #             graph_max = max(graph_max, predictions.max())

    #         graph_min, graph_max = expand(np.log(graph_min), np.log(graph_max))
    #         graph_min, graph_max = np.exp(graph_min), np.exp(graph_max)
    #         ax.set_xscale('log')
    #         ax.set_yscale('log')

    #         ax.set_xlim(graph_min, graph_max)
    #         ax.set_ylim(graph_min, graph_max)
    #         xeqy_line = ax.plot(
    #             [graph_min, graph_max], [graph_min, graph_max], color='gray', linestyle='--'
    #         )
    #         ax.set_aspect('equal')
    #         ax.set_title(f'{env}, UTD={utd} ($R^2$={r2:.2f})')
    #         ax.set_xlabel('Actual')
    #         ax.set_ylabel('Predictions')

    # color_handles = [Line2D([0], [0], color=color_map[w], lw=3) for w in critic_widths]
    # color_labels = [f'width={w}' for w in critic_widths]
    # marker_handles = [
    #     Line2D([0], [0], marker=marker_map[b], color='gray', lw=0, markersize=8)
    #     for b in batch_sizes
    # ]
    # marker_labels = [f'B={b}' for b in batch_sizes]
    # fig.legend(
    #     color_handles + marker_handles + xeqy_line,
    #     color_labels + marker_labels + ['$y=x$'],
    #     loc='center left',
    #     bbox_to_anchor=(1.05, 0.5),
    #     ncol=1,
    # )
    # fig.suptitle(metric, fontsize=16)
    # plt.tight_layout()

    # with pd.option_context(
    #     'display.max_rows',
    #     None,
    #     'display.max_columns',
    #     None,
    #     'display.width',
    #     1000,
    #     'display.expand_frame_repr',
    #     True,
    #     'display.max_colwidth',
    #     None,
    # ):
    #     print(pd.DataFrame(linear_fit)[['env', 'utd', 'formula']])
    # plt.show()

    """
    Log-linear fit
    """

    log_linear_fit = []
    fig, axes = plt.subplots(len(envs), len(utds), figsize=(3 * len(utds), 3 * len(envs)))
    axes = np.array(axes).reshape(len(envs), len(utds))

    def log_linear_featurize(data):
        X = np.array(data[['critic_width', 'batch_size']].values)
        y = np.array(data['end_of_training'].values)
        mask = (X > 0).all(axis=1) & (y > 0)
        X = X[mask]
        y = y[mask]
        if len(X) == 0:
            return None, None
        else:
            X = np.log(X)
            y = np.log(y)
            if interaction:
                X = np.column_stack([np.ones(len(X)), X, X[:, 0] * X[:, 1]])
            else:
                X = np.column_stack([np.ones(len(X)), X])
            return X, y

    for i, env in enumerate(envs):
        for j, utd in enumerate(utds):
            subset = df.query(f'env_name=="{env}" and utd=={utd}')
            if len(subset) == 0:
                continue

            mask = subset['end_of_training'] > 0
            b_corr = np.corrcoef(
                np.log(subset['batch_size'][mask]), np.log(subset['end_of_training'][mask])
            )[0, 1]
            w_corr = np.corrcoef(
                np.log(subset['critic_width'][mask]), np.log(subset['end_of_training'][mask])
            )[0, 1]
            b_w_corr = np.corrcoef(
                np.log(subset['batch_size'][mask]) * np.log(subset['critic_width'][mask]),
                np.log(subset['end_of_training'][mask]),
            )[0, 1]

            X, y = log_linear_featurize(subset)
            if X is None:
                continue
            model = sm.OLS(y, X).fit()
            if interaction:
                intercept, bs_coef, critic_width_coef, prod_coef = model.params
            else:
                intercept, bs_coef, critic_width_coef = model.params

            predictions = model.predict(X)
            loss = np.mean((predictions - y) ** 2)
            r2 = model.rsquared
            # r2 = r_squared(np.exp(y), np.exp(predictions))  # R^2 on original scale

            res_dict = {
                'env': env,
                'utd': utd,
                'intercept': intercept,
                'b_coef': bs_coef,
                'w_coef': critic_width_coef,
                'loss': loss,
                'b_corr': b_corr,
                'w_corr': w_corr,
                'formula': f'log {metric} ~ {intercept:.3e} + {bs_coef:.3e} * log B + {critic_width_coef:.3e} * log width',
            }
            if interaction:
                res_dict.update(
                    {
                        'b*w_coef': prod_coef,
                        'b*w_corr': b_w_corr,
                        'formula': f'log {metric} ~ {intercept:.3e} + {bs_coef:.3e} * log B + {critic_width_coef:.3e} * log width + {prod_coef:.3e} * log B * log width',
                    }
                )

            log_linear_fit.append(res_dict)

            ax = axes[i, j]
            graph_min, graph_max = float('inf'), float('-inf')
            for critic_width, width_group in subset.groupby('critic_width'):
                color = color_map[critic_width]
                xs = []
                ys = []
                batch_sizes = []

                for batch_size, group in width_group.groupby('batch_size'):
                    marker = marker_map[batch_size]
                    X, y = log_linear_featurize(group)
                    if X is None:
                        continue
                    predictions = model.predict(X)
                    xs.append(np.exp(y).item())
                    ys.append(np.exp(predictions).item())
                    batch_sizes.append(batch_size)

                    ax.scatter(
                        np.exp(y),
                        np.exp(predictions),
                        color=color,
                        marker=marker,
                        label=f'width={critic_width}, B={batch_size}',
                    )

                    graph_min = min(graph_min, np.exp(y).min(), np.exp(predictions).min())
                    graph_max = max(graph_max, np.exp(y).max(), np.exp(predictions).max())

                if xs:
                    idx = np.argsort(batch_sizes)
                    xs_sorted = np.array(xs)[idx]
                    ys_sorted = np.array(ys)[idx]
                    ax.plot(xs_sorted, ys_sorted, color=color, alpha=0.3)

            graph_min, graph_max = expand_log_range(graph_min, graph_max)
            ax.set_xscale('log')
            ax.set_yscale('log')

            xeqy_line = ax.plot(
                [graph_min, graph_max],
                [graph_min, graph_max],
                color='gray',
                linestyle='--',
            )
            ax.set_aspect('equal')
            ax.set_title(f'{env}, UTD={utd} ($R^2$={r2:.2f})')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predictions')

    color_handles = [Line2D([0], [0], color=color_map[w], lw=3) for w in critic_widths]
    color_labels = [f'width={w}' for w in critic_widths]
    marker_handles = [
        Line2D([0], [0], marker=marker_map[b], color='gray', lw=0, markersize=8)
        for b in batch_sizes
    ]
    marker_labels = [f'B={b}' for b in batch_sizes]
    fig.legend(
        color_handles + marker_handles + xeqy_line,
        color_labels + marker_labels + ['$y=x$'],
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        ncol=1,
    )
    fig.suptitle(metric, fontsize=16)
    plt.tight_layout()

    with pd.option_context(
        'display.max_rows',
        None,
        'display.max_columns',
        None,
        'display.width',
        1000,
        'display.expand_frame_repr',
        True,
        'display.max_colwidth',
        None,
    ):
        if interaction:
            print(
                pd.DataFrame(log_linear_fit)[
                    ['env', 'utd', 'b_corr', 'w_corr', 'b*w_corr', 'formula']
                ]
            )
        else:
            print(pd.DataFrame(log_linear_fit)[['env', 'utd', 'b_corr', 'w_corr', 'formula']])
    plt.show()


def _generic_fit_scipy(optim_f, args, init_grid, top_k=200, precise=False, parallel=False):
    """SciPy-based implementation of the generic fitting procedure"""
    if all(isinstance(s, slice) and s.start == s.stop for s in init_grid):
        brute_params = np.array([s.start for s in init_grid])
        brute_losses = np.array([optim_f(brute_params, *args)])
        init_params = brute_params.reshape(brute_params.shape[0], -1)
        init_losses = brute_losses.reshape(-1)
        top_idxs = np.argsort(init_losses)[:top_k]
        top_params = init_params[:, top_idxs]
    elif len(init_grid) <= 10:
        _, _, brute_params, brute_losses = scipy.optimize.brute(
            optim_f,
            init_grid,
            args=args,
            full_output=True,
            finish=None,
            Ns=1,
            workers=-1 if parallel else 1,
        )
        init_params = brute_params.reshape(brute_params.shape[0], -1)
        init_losses = brute_losses.reshape(-1)
        top_idxs = np.argsort(init_losses)[:top_k]
        top_params = init_params[:, top_idxs]
    else:
        num_trials = top_k * 10
        init_bounds = [
            [slice_.start, slice_.stop] if isinstance(slice_, slice) else slice_
            for slice_ in init_grid
        ]
        init_params = np.array(
            [
                [np.random.uniform(low, high) for (low, high) in init_bounds]
                for _ in range(num_trials)
            ]
        )
        init_losses = np.array([optim_f(init_point, *args) for init_point in init_params])
        top_idxs = np.argsort(init_losses)[:top_k]
        top_params = init_params[top_idxs]

    if precise:
        kw = dict(options={'ftol': 1e-12, 'gtol': 1e-12})
    else:
        kw = {}

    def helper(i):
        res = scipy.optimize.minimize(optim_f, top_params[:, i], args=args, method='L-BFGS-B', **kw)
        pred, loss = res.x, res.fun
        return pred, loss

    num_threads = int(0.5 * os.cpu_count()) if parallel else 1
    r = top_params.shape[1]
    if num_threads <= 1 or r <= num_threads:
        parallel = False

    if parallel:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            preds_losses = list(tqdm(executor.map(helper, range(r)), total=r, disable=(r == 1)))
    else:
        preds_losses = [helper(i) for i in tqdm(range(r), disable=(r == 1))]

    best_params = sorted(preds_losses, key=lambda x: x[1])[0][0]
    return best_params


def _generic_fit_jax(optim_f, args, init_grid, top_k=200, parallel=False):
    """Doesn't work yet, seems to return nan. Maybe the vmap is incorrect."""
    key = jax.random.PRNGKey(0)
    num_trials = top_k * 10
    objective = lambda p: optim_f(p, *args)

    bounds = [(s.start, s.stop) if isinstance(s, slice) else s for s in init_grid]
    dim = len(bounds)

    lows = jnp.array([b[0] for b in bounds])
    highs = jnp.array([b[1] for b in bounds])

    key, subkey = jax.random.split(key)
    rand_uniform = jax.random.uniform(subkey, (num_trials, dim))
    init_points = lows + (highs - lows) * rand_uniform

    batched_obj_fn = jax.vmap(objective)
    init_losses = batched_obj_fn(init_points)

    top_idxs = jnp.argsort(init_losses)[:top_k]
    top_points = init_points[top_idxs]

    @jax.jit
    def single_minimize(p0):
        solver = LBFGS(fun=objective)  # for whatever reason ScipyMinimize doesn't work
        opt_result = solver.run(p0)
        final_value = objective(opt_result.params)
        return opt_result.params, final_value

    results = jax.vmap(single_minimize)(top_points)
    params_list, losses_list = results
    losses_array = jnp.array(losses_list)
    best_idx = jnp.argmin(losses_array)

    return params_list[best_idx]


def _get_generic_fitter(use_jax):
    if use_jax:
        return _generic_fit_jax
    else:
        return _generic_fit_scipy


def powerlaw_fn(x, a, b, c):
    """Works with both numpy and jax"""
    return c * (1 + (x / (b + 1e-12)) ** (-a))


def powerlaw_loss(params, x, y, loss_p=2):
    a, b, c = params
    return (np.abs(powerlaw_fn(x, a, b, c) - y) ** loss_p).mean()


def fit_powerlaw(x, y, top_k=200, parallel=False, loss_p=2):
    x_scale = np.min(x)
    y_scale = np.mean(y)
    x_scaled = x / x_scale
    y_scaled = y / y_scale

    init_grid = [slice(-2, 2, 0.8), slice(0, 3, 0.6), slice(-2, 2, 0.8)]
    a, b, c = _generic_fit_scipy(
        partial(powerlaw_loss, loss_p=loss_p), (x_scaled, y_scaled), init_grid, top_k, parallel
    )
    return a, b * x_scale, c * y_scale, b, c


def powerlaw_shared_exponent_loss(params, x_scaled_data, y_scaled_data, loss_p=2):
    a = params[0]
    total_loss = 0.0

    for i, key in enumerate(x_scaled_data.keys()):
        b = params[1 + 2 * i]
        c = params[2 + 2 * i]
        x_scaled = x_scaled_data[key]
        y_scaled = y_scaled_data[key]
        pred = powerlaw_fn(x_scaled, a, b, c)
        loss = (np.abs(y_scaled - pred) ** loss_p).mean()
        total_loss += loss

    return total_loss


def _prepare_shared_data(group_data, use_jax):
    """Normalize each x by its minimum and y by its mean"""
    sample_key = list(group_data.keys())[0]
    sample_data = group_data[sample_key]
    n_features = len(sample_data) - 1
    scales = {}
    x_scaled_datas = [{} for _ in range(n_features)]
    y_scaled_data = {}

    for key in group_data:
        xs = group_data[key][:-1]
        y = group_data[key][-1]
        x_scales = [np.min(x) for x in xs]
        y_scales = np.mean(y)
        scales[key] = (*x_scales, y_scales)
        # print(xs)
        # print(x_scales)
        # print(n_features)
        for i in range(n_features):
            x_scaled_datas[i][key] = xs[i] / x_scales[i]
        y_scaled_data[key] = y / y_scales

    if use_jax:
        raise NotImplementedError
        # for key in group_data:
        #     x_scaled_datas[key] = jnp.array(x_scaled_data[key])
        #     y_scaled_data[key] = jnp.array(y_scaled_data[key])

    return (*x_scaled_datas, y_scaled_data, scales)


def fit_powerlaw_shared_exponent(
    group_data, top_k=200, use_jax=False, parallel=True, log_loss=False, loss_p=2
):
    """
    Inputs:
    group_data: dict mapping group key -> (x, y)

    Returns:
    res_dict: dict mapping group key -> (a, b, c, b_unscaled, c_unscaled) where a is shared
    """
    group_keys = list(group_data.keys())
    n_groups = len(group_keys)
    x_scaled_data, y_scaled_data, scales = _prepare_shared_data(group_data, use_jax)

    init_grid = [slice(0.5, 3.0, 0.5)] + [slice(0.1, 10.0, 3.0)] * (2 * n_groups)
    fitter = _get_generic_fitter(use_jax)
    params = fitter(
        powerlaw_shared_exponent_loss, (x_scaled_data, y_scaled_data), init_grid, top_k, parallel
    )

    a = params[0]
    res_dict = {}
    for i, key in enumerate(group_keys):
        b = params[1 + 2 * i]
        c = params[2 + 2 * i]
        x_scale, y_scale = scales[key]
        res_dict[key] = (a, b * x_scale, c * y_scale, b, c)

    return res_dict


def inverse_power_fn(x, a, b, c):
    """Works with both numpy and jax"""
    return a / (1 + (b / x) ** c)


def inverse_power_loss(params, x, y, loss_p=2):
    a, b, c = params
    return ((inverse_power_fn(x, a, b, c) - y) ** loss_p).mean()


def fit_inverse_power(x, y, top_k=200, parallel=False):
    """Fit inverse power law function to data with individual exponent"""
    x_scale = np.min(x)
    y_scale = np.mean(y)
    x_scaled = x / x_scale
    y_scaled = y / y_scale

    init_grid = [slice(0.1, 10.0, 2.0), slice(0.1, 10.0, 2.0), slice(0.1, 3.0, 0.5)]
    a, b, c = _generic_fit_scipy(
        inverse_power_loss, (x_scaled, y_scaled), init_grid, top_k, parallel
    )
    return a * y_scale, b * x_scale, c, a, b


def inverse_power_shared_exponent_loss(params, x_scaled_data, y_scaled_data, loss_p=2):
    c = params[0]
    total_loss = 0.0

    for i, key in enumerate(x_scaled_data.keys()):
        a = params[1 + 2 * i]
        b = params[2 + 2 * i]
        x_scaled = x_scaled_data[key]
        y_scaled = y_scaled_data[key]
        pred = inverse_power_fn(x_scaled, a, b, c)
        loss = (np.abs(y_scaled - pred) ** loss_p).mean()
        total_loss += loss

    return total_loss


def fit_inverse_power_shared_exponent(
    group_data, loss_p=2, top_k=200, use_jax=False, parallel=True
):
    """
    Inputs:
    group_data: dict mapping group key -> (x, y)

    Returns:
    res_dict: dict mapping group key -> (a, b, c, a_unscaled, b_unscaled) where c is shared
    """
    group_keys = list(group_data.keys())
    n_groups = len(group_keys)
    x_scaled_data, y_scaled_data, scales = _prepare_shared_data(group_data, use_jax)

    init_grid = [slice(0.5, 3.0, 0.5)] + [slice(0.1, 10.0, 4.0)] * (2 * n_groups)
    fitter = _get_generic_fitter(use_jax)
    params = fitter(
        partial(inverse_power_shared_exponent_loss, loss_p=loss_p),
        (x_scaled_data, y_scaled_data),
        init_grid,
        top_k,
        parallel,
    )

    c = params[0]
    res_dict = {}
    for i, key in enumerate(group_keys):
        a = params[1 + 2 * i]
        b = params[2 + 2 * i]
        x_scale, y_scale = scales[key]
        res_dict[key] = (a * y_scale, b * x_scale, c, a, b)

    return res_dict


def _two_var_loss_fn_helper(predict_fn, raw, x1, x2, y, loss_p=2):
    u, v, s, t = raw
    # a = np.exp(u)
    # b = np.exp(v)
    a = softplus(u)
    b = softplus(v)
    c = softplus(s)
    alpha = softplus(t)
    return (np.abs(predict_fn(x1, x2, a, b, c, alpha) - y) ** loss_p).mean()


def _two_var_log_loss_fn_helper(predict_fn, raw, x1, x2, y, loss_p=2):
    u, v, s, t = raw
    # a = np.exp(u)
    # b = np.exp(v)
    a = softplus(u)
    b = softplus(v)
    c = softplus(s)
    alpha = softplus(t)
    return (np.abs(np.log(predict_fn(x1, x2, a, b, c, alpha)) - np.log(y)) ** loss_p).mean()


def _get_two_var_log_loss_fn_helper(log_fit, loss_p=2):
    return (
        partial(_two_var_log_loss_fn_helper, loss_p=loss_p)
        if log_fit
        else partial(_two_var_loss_fn_helper, loss_p=loss_p)
    )


def _fit_two_coef_two_exp(loss_fn, x1, x2, y, top_k=200, parallel=False):
    x1_scale = np.min(x1)
    x2_scale = np.min(x2)
    y_scale = np.mean(y)
    x1_scaled = x1 / x1_scale
    x2_scaled = x2 / x2_scale
    y_scaled = y / y_scale

    # init_grid = [
    #     slice(0.1, 10.0, 2.0),
    #     slice(0.1, 10.0, 2.0),
    #     slice(0.1, 3.0, 0.5),
    #     slice(0.1, 3.0, 0.5),
    # ]
    init_grid = [slice(0.0, 0.0, 1.0) for _ in range(4)]
    raw = _generic_fit_scipy(loss_fn, (x1_scaled, x2_scaled, y_scaled), init_grid, top_k, parallel)
    u, v, s, t = raw
    # a = np.exp(u)
    # b = np.exp(v)
    # c = np.exp(w)
    a = softplus(u)
    b = softplus(v)
    c = softplus(s)
    alpha = softplus(t)
    return (a, b, c, alpha), (x1_scale, x2_scale, y_scale)


def _fit_two_coef_two_exp_log_normalize(loss_fn, x1, x2, y, top_k=200, parallel=False):
    x1_scaled, x1_m, x1_s = _log_rescale(x1)
    x2_scaled, x2_m, x2_s = _log_rescale(x2)
    y_scale = np.mean(y)
    y_scaled = y / y_scale

    # init_grid = [
    #     slice(0.1, 10.0, 2.0),
    #     slice(0.1, 10.0, 2.0),
    #     slice(0.1, 3.0, 0.5),
    #     slice(0.1, 3.0, 0.5),
    # ]
    init_grid = [slice(0.0, 0.0, 1.0) for _ in range(4)]
    raw = _generic_fit_scipy(loss_fn, (x1_scaled, x2_scaled, y_scaled), init_grid, top_k, parallel)
    u, v, s, t = raw
    a = softplus(u)  # coefficient
    b = softplus(v)  # coefficient
    c = softplus(s)  # exponent
    alpha = softplus(t)  # exponent

    return (a, b, c, alpha), (x1_m, x1_s, x2_m, x2_s, y_scale)


def inverse_power_product_fn(x1, x2, a, b, c, alpha):
    """x1=sigma, x2=N"""
    return a / (x1**alpha * (1 + (b / x2) ** c))


def fit_inverse_power_product(x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2):
    loss = partial(_get_two_var_log_loss_fn_helper(log_loss, loss_p), inverse_power_product_fn)
    (a, b, c, alpha), (x1_scale, x2_scale, y_scale) = _fit_two_coef_two_exp(
        loss, x1, x2, y, top_k, parallel
    )
    return a * y_scale / (x1_scale**alpha), b * x2_scale, c, alpha, a, b


def fit_inverse_power_product_log_normalize(
    x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2
):
    loss = partial(_get_two_var_log_loss_fn_helper(log_loss, loss_p), inverse_power_product_fn)
    (a, b, c, alpha), (x1_m, x1_s, x2_m, x2_s, y_scale) = _fit_two_coef_two_exp_log_normalize(
        loss, x1, x2, y, top_k, parallel
    )
    return (
        a * y_scale * np.exp(alpha * x1_m / x1_s),
        b**x2_s * np.exp(x2_m),
        c / x2_s,
        alpha / x1_s,
        a,
        b,
        c,
    )


def inverse_power_product_flip_fn(x1, x2, a, b, c, alpha):
    """x1=sigma, x2=N"""
    return (a * x2**c) / (1 + b * (x1**alpha))


def fit_inverse_power_flip_product(x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2):
    loss = partial(_get_two_var_log_loss_fn_helper(log_loss, loss_p), inverse_power_product_flip_fn)
    (a, b, c, alpha), (x1_scale, x2_scale, y_scale) = _fit_two_coef_two_exp(
        loss, x1, x2, y, top_k, parallel
    )
    return a * y_scale / (x2_scale**c), b / (x1_scale**alpha), c, alpha, a, b


def inverse_power_product_numerator_fn(x1, x2, a, b, c, alpha):
    """x1=sigma, x2=N"""
    return a * x1 ** (-alpha) * (1 + (x2 / b) ** c)


def fit_inverse_power_numerator_product(
    x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2
):
    loss = partial(
        _get_two_var_log_loss_fn_helper(log_loss, loss_p), inverse_power_product_numerator_fn
    )
    (a, b, c, alpha), (x1_scale, x2_scale, y_scale) = _fit_two_coef_two_exp(
        loss, x1, x2, y, top_k, parallel
    )
    return a * y_scale * (x1_scale**alpha), b * x2_scale, c, alpha, a, b


def inverse_power_product_numerator_flip_fn(x1, x2, a, b, c, alpha):
    """x1=sigma, x2=N"""
    return a * x2**c * (1 + (b / x1) ** alpha)


def fit_inverse_power_numerator_flip_product(
    x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2
):
    loss = partial(
        _get_two_var_log_loss_fn_helper(log_loss, loss_p), inverse_power_product_numerator_flip_fn
    )
    (a, b, c, alpha), (x1_scale, x2_scale, y_scale) = _fit_two_coef_two_exp(
        loss, x1, x2, y, top_k, parallel
    )
    return a * y_scale / (x2_scale**c), b * x1_scale, c, alpha, a, b


def denominator_sum_power_fn(x1, x2, a, b, c, alpha):
    """x1=sigma, x2=N"""
    return a / (x1**alpha + b * x2 ** (-c))


def fit_denominator_sum_power(x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2):
    loss = partial(_get_two_var_log_loss_fn_helper(log_loss, loss_p), denominator_sum_power_fn)
    (a, b, c, alpha), (x1_scale, x2_scale, y_scale) = _fit_two_coef_two_exp(
        loss, x1, x2, y, top_k, parallel
    )
    return a * y_scale * (x1_scale**alpha), b * x2_scale**c * (x1_scale**alpha), c, alpha, a, b


def sum_power_fn(x1, x2, a, b, c, alpha):
    """x1=sigma, x2=N"""
    return a * (b * x1 ** (-alpha) + x2**c)


def fit_sum_power(x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2):
    loss = partial(_get_two_var_log_loss_fn_helper(log_loss, loss_p), denominator_sum_power_fn)
    (a, b, c, alpha), (x1_scale, x2_scale, y_scale) = _fit_two_coef_two_exp(
        loss, x1, x2, y, top_k, parallel
    )
    return a * y_scale / (x2_scale**c), b * x1_scale**alpha * x2_scale**c, c, alpha, a, b


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sum_of_powerlaw_fn(x1, x2, a, alpha, b, beta, c):
    return c * (1 + (a / x1) ** alpha + (b / x2) ** beta)


def sum_of_powerlaw_loss(raw, x1, x2, y, loss_p=2):
    u, s, v, t, w = raw
    # a = np.exp(u)
    # b = np.exp(v)
    # c = np.exp(w)
    a = softplus(u)
    b = softplus(v)
    c = softplus(w)
    alpha = softplus(s)
    beta = softplus(t)
    return (np.abs(sum_of_powerlaw_fn(x1, x2, a, alpha, b, beta, c) - y) ** loss_p).mean()


def sum_of_powerlaw_log_loss(raw, x1, x2, y, loss_p=2):
    u, s, v, t, w = raw
    # a = np.exp(u)
    # b = np.exp(v)
    # c = np.exp(w)
    a = softplus(u)
    b = softplus(v)
    c = softplus(w)
    alpha = softplus(s)
    beta = softplus(t)
    return (
        np.abs(np.log(sum_of_powerlaw_fn(x1, x2, a, alpha, b, beta, c)) - np.log(y)) ** loss_p
    ).mean()


def _log_rescale(x, lo=0.5, hi=2.0):
    """Normalize x via exp((log x - m)/s) to [lo, hi]. Returns (x_scaled, m, s)."""
    x_min, x_max = min(x), max(x)
    s = (np.log(x_max) - np.log(x_min)) / (np.log(hi) - np.log(lo))
    m = np.log(x_min) - s * np.log(lo)
    x_scaled = np.exp((np.log(x) - m) / s)
    return x_scaled, m, s


def _log_rescale_inverse(x_scaled, m, s):
    return np.exp(s * np.log(x_scaled) + m)


def fit_sum_of_powerlaw(
    x1, x2, y, top_k=500, parallel=True, log_loss=False, loss_p=2, init_grid=None, init_p=None
):
    assert init_grid is None or init_p is None, 'cannot specify both init_grid and init_p'

    x1_scaled, x1_m, x1_s = _log_rescale(x1)
    x2_scaled, x2_m, x2_s = _log_rescale(x2)
    y_scale = np.mean(y)
    y_scaled = y / y_scale
    # print('blah')
    # print('x1', x1_scaled)
    # print('x2', x2_scaled)
    # print('y', y_scaled)

    # if init_grid is None:
    #     init_grid = [
    #         slice(0.1, 2.1, 0.5),
    #         slice(1.0, 2.5, 0.5),
    #         slice(0.1, 2.1, 0.5),
    #         slice(1.0, 2.5, 0.5),
    #         slice(0.05, 1.0, 0.2),
    #     ]
    # if init_p is not None:
    #     assert len(init_p) == 5
    #     init_grid = [(0.5*p, p, 2*p) for p in init_p]
    init_grid = [slice(0.0, 0.0, 1.0) for _ in range(5)]

    loss_fn = (
        partial(sum_of_powerlaw_log_loss, loss_p=loss_p)
        if log_loss
        else partial(sum_of_powerlaw_loss, loss_p=loss_p)
    )
    raw = _generic_fit_scipy(loss_fn, (x1_scaled, x2_scaled, y_scaled), init_grid, top_k, parallel)
    u, s, v, t, w = raw
    # a = np.exp(u)
    # b = np.exp(v)
    # c = np.exp(w)
    a = softplus(u)
    b = softplus(v)
    c = softplus(w)
    alpha = softplus(s)
    beta = softplus(t)

    return (
        a**x1_s * np.exp(x1_m),
        alpha / x1_s,
        b**x2_s * np.exp(x2_m),
        beta / x2_s,
        c * y_scale,
        alpha,
        beta,
        a,
        b,
        c,
    )


def sum_of_powerlaw_shared_exponent_loss(
    params, x1_scaled_data, x2_scaled_data, y_scaled_data, loss_p=2
):
    # alpha, beta = params[0], params[1]
    s, t = params[0], params[1]
    total_loss = 0.0

    for i, key in enumerate(x1_scaled_data.keys()):
        # a = params[2 + 3 * i]
        # b = params[3 + 3 * i]
        # c = params[4 + 3 * i]
        u, v, w = params[2 + 3 * i : 5 + 3 * i]
        x1_scaled = x1_scaled_data[key]
        x2_scaled = x2_scaled_data[key]
        y_scaled = y_scaled_data[key]
        # a = np.exp(u)
        # b = np.exp(v)
        # c = np.exp(w)
        a = softplus(u)
        b = softplus(v)
        c = softplus(w)
        alpha = softplus(s)
        beta = softplus(t)
        pred = sum_of_powerlaw_fn(x1_scaled, x2_scaled, a, alpha, b, beta, c)
        loss = (np.abs(y_scaled - pred) ** loss_p).mean()
        total_loss += loss

    return total_loss


def sum_of_powerlaw_shared_exponent_log_loss(
    params, x1_scaled_data, x2_scaled_data, y_scaled_data, loss_p=2
):
    # alpha, beta = params[0], params[1]
    s, t = params[0], params[1]
    total_loss = 0.0

    for i, key in enumerate(x1_scaled_data.keys()):
        # a = params[2 + 3 * i]
        # b = params[3 + 3 * i]
        # c = params[4 + 3 * i]
        u, v, w = params[2 + 3 * i : 5 + 3 * i]
        x1_scaled = x1_scaled_data[key]
        x2_scaled = x2_scaled_data[key]
        y_scaled = y_scaled_data[key]
        # a = np.exp(u)
        # b = np.exp(v)
        # c = np.exp(w)
        a = softplus(u)
        b = softplus(v)
        c = softplus(w)
        alpha = softplus(s)
        beta = softplus(t)
        pred = sum_of_powerlaw_fn(x1_scaled, x2_scaled, a, alpha, b, beta, c)
        loss = (np.abs(np.log(y_scaled) - np.log(pred)) ** loss_p).mean()
        total_loss += loss

    return total_loss


def fit_sum_of_powerlaw_shared_exponent(
    group_data, top_k=200, log_loss=False, loss_p=2, use_jax=False, parallel=True
):
    group_keys = list(group_data.keys())
    n_groups = len(group_keys)
    x1_scaled_data = {}
    x2_scaled_data = {}
    y_scaled_data = {}

    random_key = np.random.choice(group_keys)
    x1, x2, y = group_data[random_key]
    _, x1_m, x1_s = _log_rescale(x1)
    _, x2_m, x2_s = _log_rescale(x2)
    y_scale = np.mean(y)

    for key in group_data:
        x1, x2, y = group_data[key]
        x1_scaled_data[key] = np.exp((np.log(x1) - x1_m) / x1_s)
        x2_scaled_data[key] = np.exp((np.log(x2) - x2_m) / x2_s)
        y_scaled_data[key] = y / y_scale

    init_grid = [slice(0.0, 0.0, 1.0) for _ in range(2 + 3 * n_groups)]
    loss_fn = (
        partial(sum_of_powerlaw_shared_exponent_log_loss, loss_p=loss_p)
        if log_loss
        else partial(sum_of_powerlaw_shared_exponent_loss, loss_p=loss_p)
    )
    params = _generic_fit_scipy(
        loss_fn, (x1_scaled_data, x2_scaled_data, y_scaled_data), init_grid, top_k, parallel
    )

    # alpha, beta = params[0], params[1]
    s, t = params[0], params[1]
    res_dict = {}
    for i, key in enumerate(group_keys):
        # a = params[2 + 3 * i]
        # b = params[3 + 3 * i]
        # c = params[4 + 3 * i]
        u, v, w = params[2 + 3 * i : 5 + 3 * i]
        # a = np.exp(u)
        # b = np.exp(v)
        # c = np.exp(w)
        a = softplus(u)
        b = softplus(v)
        c = softplus(w)
        alpha = softplus(s)
        beta = softplus(t)
        res_dict[key] = (
            a**x1_s * np.exp(x1_m),
            alpha / x1_s,
            b**x2_s * np.exp(x2_m),
            beta / x2_s,
            c * y_scale,
            alpha,
            beta,
            a,
            b,
            c,
        )

    return res_dict


def sum_of_powerlaw_shift_fn(x1, x2, a, alpha, b, beta, gamma, c):
    return c * (1 + (a / x1) ** alpha + (b * (x1**gamma) / x2) ** beta)


def sum_of_powerlaw_shift_loss(params, x1, x2, y, loss_p=2):
    a, alpha, b, beta, gamma, c = params
    return (
        np.abs(sum_of_powerlaw_shift_fn(x1, x2, a, alpha, b, beta, gamma, c) - y) ** loss_p
    ).mean()


def sum_of_powerlaw_shift_log_loss(params, x1, x2, y, loss_p=2):
    a, alpha, b, beta, gamma, c = params
    return (
        np.abs(np.log(sum_of_powerlaw_shift_fn(x1, x2, a, alpha, b, beta, gamma, c)) - np.log(y))
        ** loss_p
    ).mean()


def fit_sum_of_powerlaw_shift(x1, x2, y, top_k=200, parallel=True, log_loss=False, loss_p=2):
    x1_scale = np.min(x1)
    x2_scale = np.min(x2)
    y_scale = np.mean(y)
    x1_scaled = x1 / x1_scale
    x2_scaled = x2 / x2_scale
    y_scaled = y / y_scale

    init_grid = [
        slice(0.1, 5.0, 0.8),
        slice(0.1, 3.0, 0.5),
        slice(0.1, 5.0, 0.8),
        slice(0.1, 3, 0.5),
        slice(0.1, 3, 0.5),
        slice(0.1, 5.0, 0.8),
    ]
    loss_fn = (
        partial(sum_of_powerlaw_shift_log_loss, loss_p=loss_p)
        if log_loss
        else partial(sum_of_powerlaw_shift_loss, loss_p=loss_p)
    )
    params = _generic_fit_scipy(
        loss_fn, (x1_scaled, x2_scaled, y_scaled), init_grid, top_k, parallel
    )
    a, alpha, b, beta, gamma, c = params
    return (
        a * x1_scale,
        alpha,
        b * x2_scale / (x1_scale**gamma),
        beta,
        gamma,
        c * y_scale,
        a,
        b,
        c,
    )


# def fit_sum_of_powerlaw_shared(group_data, top_k=200, parallel=True):
#     """
#     Inputs:
#     group_data: dict mapping group key -> (x1, x2, y), e.g. (utd, critic_width, data)

#     Output:
#     res_dict: dict mapping group key -> (a, alpha, b, beta, c, c_unscaled)
#     """
#     group_keys = list(group_data.keys())
#     x1_scaled, x2_scaled, y_scaled, scales = _prepare_shared_data(group_data, use_jax=False)

#     init_grid = [
#         slice(0.1, 5.0, 0.8), slice(0.1, 3.0, 0.5), slice(0.1, 5.0, 0.8), slice(0.1, 3, 0.5), slice(0.1, 5.0, 0.8)
#     ]
#     params = _generic_fit_scipy(
#         sum_of_powerlaw_loss, (x1_scaled, x2_scaled, y_scaled), init_grid, top_k, parallel
#     )
#     a, alpha, b, beta, c = params
#     res_dict = {}
#     for key in group_keys:
#         x_scale, y_scale = scales[key]
#         res_dict[key] = (a * x_scale[:, 0], alpha, b * x_scale[:, 1], beta, c * y_scale, c)
#     return res_dict
