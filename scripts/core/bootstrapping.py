import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict

from rliable import plot_utils as rliable_plot_utils
from qscaled.utils import plot_utils as qscaled_plot_utils
from qscaled.core.preprocessing import get_envs, get_utds

from scripts.utils import abbreviate_number, expand_log_range
from scripts.core.fitting import (
    fit_powerlaw,
    powerlaw_fn,
    fit_powerlaw_shared_exponent,
    fit_inverse_power,
    inverse_power_fn,
    fit_inverse_power_shared_exponent,
    sum_of_powerlaw_fn,
    fit_sum_of_powerlaw,
    sum_of_powerlaw_shift_fn,
    fit_sum_of_powerlaw_shift,
    inverse_power_product_fn,
    fit_inverse_power_product,
    fit_inverse_power_product_log_normalize,
    inverse_power_product_flip_fn,
    fit_inverse_power_flip_product,
    inverse_power_product_numerator_fn,
    fit_inverse_power_numerator_product,
    inverse_power_product_numerator_flip_fn,
    fit_inverse_power_numerator_flip_product,
    denominator_sum_power_fn,
    fit_denominator_sum_power,
    sum_power_fn,
    fit_sum_power,
    fit_sum_of_powerlaw_shared_exponent,
)


this_file = os.path.abspath(__file__)
fits_save_dir = os.path.join(os.path.dirname(this_file), '../experiments/fits')
os.makedirs(fits_save_dir, exist_ok=True)


def compute_time_to_threshold(df, param_name, threshold_idx=-1):
    """Precompute time to threshold and uncertainty for each group."""
    assert param_name in ['lr', 'bs']
    param_key = 'learning_rate' if param_name == 'lr' else 'batch_size'
    group_key = 'batch_size' if param_name == 'lr' else 'learning_rate'

    grouped = df.groupby(['env_name', 'utd', 'critic_width', group_key])

    results = []
    for (env, utd, critic_width, group_value), group in grouped:
        param_groups = group.groupby(param_key)

        time_to_threshold = param_groups.apply(
            lambda x: x['crossings'].iloc[0][threshold_idx]
        ).dropna()

        time_to_threshold_bootstrap = param_groups.apply(
            lambda x: x['crossings_bootstrap'].iloc[0][:, threshold_idx]
        )

        results.append(
            {
                'env_name': env,
                'utd': utd,
                'critic_width': critic_width,
                group_key: group_value,
                'critic_params': int(group['critic_params'].mean()),
                'param_values': np.array(time_to_threshold_bootstrap.index),
                'times': np.array(
                    time_to_threshold.reindex(time_to_threshold_bootstrap.index, fill_value=np.inf)
                ),
                'times_bootstrap': np.array(time_to_threshold_bootstrap.tolist()),
            }
        )

    return pd.DataFrame(results)


def _grid_best_uncertainty(df, param_name, threshold_idx, print_pivot):
    """
    Almost identical to the same method from qscaled, but now also grouped by model size.

    Make and print a table with uncertainty-corrected best
    param_name = learning rate (batch size) for each
    environment, batch size (learning rate), and UTD with environment as rows
    and utd x batch size (learning rate) as columns.

    This description is somewhat confusing; the docstrings for
    `grid_best_uncertainty_lr` and `grid_best_uncertainty_bs` are more clear.
    """
    assert param_name in ['lr', 'bs']

    if param_name == 'lr':
        param_key = 'learning_rate'
        group_key = 'batch_size'
    else:
        param_key = 'batch_size'
        group_key = 'learning_rate'

    grouped = df.groupby(['env_name', 'utd', 'critic_width', group_key])

    # Find best learning rate (batch size) for each group
    results = []
    for (env, utd, critic_width, group_value), group in grouped:
        # Time to hit thresholds[threshold_idx]
        param_groups = group.groupby(param_key)
        time_to_threshold = param_groups.apply(lambda x: x['crossings'].iloc[0][threshold_idx])
        time_to_threshold_std = param_groups.apply(
            lambda x: x['crossings_std'].iloc[0][threshold_idx]
        )
        idx = time_to_threshold.notna()
        time_to_threshold_std = time_to_threshold_std.fillna(0)
        time_to_threshold = time_to_threshold[idx]
        time_to_threshold_std = time_to_threshold_std[idx]

        good_param_values = np.array(time_to_threshold.index)
        good_param_groups = group[group[param_key].isin(good_param_values)].groupby(param_key)

        # Get bootstrap samples
        time_to_threshold_bootstrap = good_param_groups.apply(
            lambda x: x['crossings_bootstrap'].iloc[0][:, threshold_idx]
        )
        times_bootstrap = np.array(time_to_threshold_bootstrap.tolist())

        # Find best learning rate (batch size)
        times_bootstrap_inf = np.where(np.isnan(times_bootstrap), np.inf, times_bootstrap)
        best_value_bootstrap = good_param_values[np.argmin(times_bootstrap_inf, axis=0)]
        min_time_bootstrap = np.min(times_bootstrap_inf, axis=0)

        best_value_bootstrap_mean = np.mean(best_value_bootstrap)
        best_value_log_dist = np.abs(np.log(good_param_values) - np.log(best_value_bootstrap_mean))
        best_value_rounded = good_param_values[best_value_log_dist.argsort()[0]]

        if len(time_to_threshold) > 0:
            best_value = time_to_threshold.idxmin(skipna=True)
            min_time = time_to_threshold[best_value]
            min_time_rounded = time_to_threshold[best_value_rounded]
        else:
            best_value = float('nan')
            min_time = float('inf')
            min_time_rounded = float('inf')

        # Solution 1: overlapping errorbars
        try:
            param_good_enough = (time_to_threshold - time_to_threshold[best_value]) <= np.sqrt(
                (time_to_threshold_std[best_value]) ** 2 + time_to_threshold_std**2
            )
        except:
            print(time_to_threshold)
            print(time_to_threshold_std)
            print(best_value)
            raise
        good_enough_params = param_good_enough.index[param_good_enough]
        smallest_good_enough_param = good_enough_params.min()
        is_smallest_param = smallest_good_enough_param == min(good_param_values)
        largest_good_enough_param = good_enough_params.max()
        is_largest_param = largest_good_enough_param == max(good_param_values)

        # Solution 2: bootstrapped confidence interval
        # best_value_time_bootstrap = times_bootstrap_inf[param_values.tolist().index(best_value)]
        # np.seterr(all='ignore')
        # time_diff_bootstrap = times_bootstrap_inf - best_value_time_bootstrap
        # np.seterr(all='warn')
        # good_enough = (time_diff_bootstrap > 0).mean(axis=1) <= data_thresh
        # # if good_enough.sum() == 0:
        # #     print(time_diff_bootstrap)
        # #     print((time_diff_bootstrap > 0).mean(axis=1))
        # #     print(good_enough)
        # if env == 'h1-crawl-v0' and critic_width==512 and utd==2:
        #     print((time_diff_bootstrap > 0).mean(axis=1))
        # largest_good_enough_param = np.max(param_values[good_enough])
        # is_largest_param = (largest_good_enough_param == max(param_values))

        results.append(
            {
                'env_name': env,
                group_key: group_value,  # batch size (learning rate)
                'utd': utd,
                'critic_width': critic_width,
                'critic_params': int(group['critic_params'].mean()),
                f'best_{param_name}': best_value,
                'time_to_threshold': min_time,
                f'best_{param_name}_bootstrap': best_value_bootstrap,
                f'best_{param_name}_bootstrap_mean': np.mean(best_value_bootstrap),
                f'best_{param_name}_bootstrap_mean_log': np.exp(
                    np.log(best_value_bootstrap).mean()
                ),
                f'best_{param_name}_bootstrap_mean_rounded': best_value_rounded,
                'time_to_threshold_rounded': min_time_rounded,
                f'best_{param_name}_bootstrap_std': np.std(best_value_bootstrap),
                'time_to_threshold_bootstrap': min_time_bootstrap,
                f'smallest_{param_name}_good_enough': smallest_good_enough_param,
                f'good_enough_{param_name}_is_smallest': is_smallest_param,
                f'largest_{param_name}_good_enough': largest_good_enough_param,
                f'good_enough_{param_name}_is_largest': is_largest_param,
            }
        )

    df_best = pd.DataFrame(results)

    if print_pivot:
        pd.set_option('display.float_format', '{:.1e}'.format)

        pivot_df = df_best.pivot_table(
            index='utd',
            columns=['env_name', group_key],
            values=f'best_{param_name}',
            aggfunc='first',
        )
        print(f'\nBest {param_key}:')
        print(pivot_df.to_string())

        pivot_df_bootstrap = df_best.pivot_table(
            index='utd',
            columns=['env_name', group_key],
            values=f'best_{param_name}_bootstrap',
            aggfunc=lambda x: np.mean(np.stack(x)),
        )
        print(f'\nUncertainty-Corrected Best {param_key}:')
        print(pivot_df_bootstrap.to_string())

    return df_best


def grid_best_uncertainty_lr(df, threshold_idx=-1, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best learning rate for
    each environment, batch size, and UTD with environment as rows and
    utd x batch size as columns.
    """
    return _grid_best_uncertainty(df, 'lr', threshold_idx, print_pivot)


def grid_best_uncertainty_bs(df, threshold_idx=-1, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best batch size for
    each environment, learning rate, and UTD with environment as rows and
    utd x learning rate as columns.
    """
    return _grid_best_uncertainty(df, 'bs', threshold_idx, print_pivot)


def print_optimal_hparams(best_bs, best_lr):
    if best_bs is not None:
        print('Empirical optimal batch size:')
        print(best_bs.pivot(index='utd', columns=['env_name', 'critic_width'], values='best_bs'))
        print()

        print('Uncertainty-corrected optimal batch size:')
        print(
            best_bs.pivot(
                index='utd',
                columns=['env_name', 'critic_width'],
                values='best_bs_bootstrap_mean',
            )
        )
        print()

    if best_lr is not None:
        print('Empirical optimal learning rate:')
        print(best_lr.pivot(index='utd', columns=['env_name', 'critic_width'], values='best_lr'))
        print()

        print('Uncertainty-corrected optimal learning rate:')
        print(
            best_lr.pivot(
                index='utd',
                columns=['env_name', 'critic_width'],
                values='best_lr_bootstrap_mean',
            )
        )
        print()


def build_log_features(df, xcols, interaction=False):
    """Build log-transformed feature matrix, optionally with interaction term."""
    X = np.log(df[xcols].values)
    if interaction:
        interaction_term = X[:, 0] * X[:, 1]
        X = np.column_stack([X, interaction_term])
    return sm.add_constant(X)


def fit_log_linear_helper(df, xcols, ycol, interaction=False):
    """Fit a log-linear model with interaction: log(F) ~ log(sigma), log(N), log(sigma)*log(N)"""
    assert len(xcols) == 2, 'Only two features are supported for interaction terms'
    intercepts = {}
    slopes = {}
    r2s = {}

    for env, group in df.groupby('env_name'):
        X = build_log_features(group, xcols, interaction)
        y = np.log(group[ycol].values)
        idx = (~np.isnan(y)) & (y < np.inf) & (y > -np.inf)
        y = y[idx]
        X = X[idx]
        model = sm.OLS(y, X).fit()

        intercept, slope = model.params[0], model.params[1:]
        r2 = model.rsquared
        intercepts[env] = intercept
        slopes[env] = slope
        r2s[env] = r2

        fit_str = (
            f'{env}: log {ycol} ~ {intercept:.4e} + '
            f'{slope[0]:.4f} * log {xcols[0]} + {slope[1]:.4f} * log {xcols[1]}'
        )
        if interaction:
            fit_str += f' + {slope[2]:.4e} * log {xcols[0]} * log {xcols[1]}'
        print(fit_str)

    return model, slopes, intercepts, r2s


def fit_log_linear(df, xcols, ycol):
    return fit_log_linear_helper(df, xcols, ycol, interaction=False)


def fit_log_linear_interaction(df, xcols, ycol):
    return fit_log_linear_helper(df, xcols, ycol, interaction=True)


def _fit_log_linear_shared_slope_helper(df, xcols, ycol, interaction=False):
    """Fit a log-linear model with shared slope across environments, optionally with interaction term."""
    assert len(xcols) == 2, 'Only two features are supported for interaction terms'

    envs = sorted(df['env_name'].unique())
    env_dummies = pd.get_dummies(df['env_name'], drop_first=False)
    log_features = np.log(df[xcols].values)

    if interaction:
        interaction_term = (log_features[:, 0] * log_features[:, 1])[:, None]
        log_features = np.hstack([log_features, interaction_term])

    X = np.hstack([env_dummies.values, log_features])
    y = np.log(df[ycol].values)

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    num_envs = len(envs)
    intercept = model.params[0]
    env_coefs = model.params[1 : 1 + num_envs]
    shared_slope = model.params[1 + num_envs :]
    r2 = model.rsquared

    env_intercepts = {env: intercept + env_coefs[i] for i, env in enumerate(envs)}

    for env, b in env_intercepts.items():
        line = f'{env}: log {ycol} ~ {b:.4e}'
        for i, name in enumerate(xcols):
            line += f' + {shared_slope[i]:.4f} * log {name}'
        if interaction:
            line += f' + {shared_slope[2]:.4e} * log {xcols[0]} * log {xcols[1]}'
        print(line)

    return model, shared_slope, env_intercepts, r2


def fit_log_linear_shared_slope(df, xcols, ycol):
    return _fit_log_linear_shared_slope_helper(df, xcols, ycol, interaction=False)


def fit_log_linear_shared_slope_interaction(df, xcols, ycol):
    return _fit_log_linear_shared_slope_helper(df, xcols, ycol, interaction=True)


def predict_log_linear(df, xcols, slope, intercept):
    """Predict using the log-linear model."""
    assert len(xcols) == 2, 'For now'
    df_slopes = np.stack(df['env_name'].map(slope).values)
    df_intercepts = df['env_name'].map(intercept).values
    x = df[xcols].values
    return np.exp(df_intercepts + (np.log(x) * df_slopes).sum(axis=1))


def predict_log_linear_interaction(df, xcols, slope, intercept):
    """Predict using the log-linear model."""
    assert len(xcols) == 2, 'For now'
    x = df[xcols].values
    x = np.log(x)
    x = np.column_stack([x, x[:, 0] * x[:, 1]])
    return np.exp(intercept + x @ slope)


def predict_log_linear_shared_slope(df, xcols, shared_slope, env_intercepts):
    """Predict using the log-linear model. (utd, critic_params)"""
    assert len(xcols) == 2, 'For now'
    x = df[xcols].values
    intercepts = df['env_name'].map(env_intercepts).values
    return np.exp(intercepts + np.log(x) @ shared_slope)


def predict_log_linear_shared_slope_interaction(df, xcols, shared_slope, env_intercepts):
    """Predict using the log-linear model. (utd, critic_params)"""
    assert len(xcols) == 2, 'For now'
    x = df[xcols].values
    x = np.log(x)
    x = np.column_stack([x, x[:, 0] * x[:, 1]])
    intercepts = df['env_name'].map(env_intercepts).values
    return np.exp(intercepts + x @ shared_slope)


def predict_powerlaw(df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, b, c, b_unscaled, c_unscaled = params_dict[(env, row[xcols[0]])]
        predictions.append(powerlaw_fn(row[xcols[1]], a, b, c))
    return np.array(predictions)


def predict_powerlaw_shared_exponent(df, xcols, params_dict):
    assert xcols == ['utd', 'critic_params'], 'For now'
    return predict_powerlaw(df, xcols, params_dict)


def predict_inverse_power(df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, b, c, a_unscaled, b_unscaled = params_dict[(env, row[xcols[0]])]
        predictions.append(inverse_power_fn(row[xcols[1]], a, b, c))
    return np.array(predictions)


def predict_inverse_power_shared_exponent(df, xcols, params_dict):
    assert xcols == ['utd', 'critic_params'], 'For now'
    return predict_inverse_power(df, xcols, params_dict)


def predict_sum_of_powerlaw(df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, alpha, b, beta, c, alpha_unscaled, beta_unscaled, a_unscaled, b_unscaled, c_unscaled = (
            params_dict[env]
        )
        predictions.append(sum_of_powerlaw_fn(row[xcols[0]], row[xcols[1]], a, alpha, b, beta, c))
    return np.array(predictions)


def predict_sum_of_powerlaw_shift(df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, alpha, b, beta, gamma, c, a_unscaled, b_unscaled, c_unscaled = params_dict[env]
        predictions.append(
            sum_of_powerlaw_shift_fn(row[xcols[0]], row[xcols[1]], a, alpha, b, beta, gamma, c)
        )
    return np.array(predictions)


def predict_sum_of_powerlaw_shared_exponent(df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, alpha, b, beta, c, alpha_unscaled, beta_unscaled, a_unscaled, b_unscaled, c_unscaled = (
            params_dict[env]
        )
        predictions.append(sum_of_powerlaw_fn(row[xcols[0]], row[xcols[1]], a, alpha, b, beta, c))
    return np.array(predictions)


def _bivariate_twocoef_twoexp(predict_fn, df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, b, c, alpha, a_unscaled, b_unscaled = params_dict[env]
        predictions.append(predict_fn(row[xcols[0]], row[xcols[1]], a, b, c, alpha))
    return np.array(predictions)


def _bivariate_twocoef_twoexp_log_normalize(predict_fn, df, xcols, params_dict):
    assert len(xcols) == 2
    predictions = []
    for _, row in df.iterrows():
        env = row['env_name']
        a, b, c, alpha, a_unscaled, b_unscaled, c_unscaled = params_dict[env]
        predictions.append(predict_fn(row[xcols[0]], row[xcols[1]], a, b, c, alpha))
    return np.array(predictions)


def predict_inverse_power_product(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp(inverse_power_product_fn, df, xcols, params_dict)


def predict_inverse_power_product_log_normalize(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp_log_normalize(inverse_power_product_fn, df, xcols, params_dict)


def predict_inverse_power_flip_product(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp(inverse_power_product_flip_fn, df, xcols, params_dict)


def predict_inverse_power_numerator_product(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp(inverse_power_product_numerator_fn, df, xcols, params_dict)


def predict_inverse_power_numerator_flip_product(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp(
        inverse_power_product_numerator_flip_fn, df, xcols, params_dict
    )


def predict_denominator_sum_power(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp(denominator_sum_power_fn, df, xcols, params_dict)


def predict_sum_power(df, xcols, params_dict):
    return _bivariate_twocoef_twoexp(sum_power_fn, df, xcols, params_dict)


def insert_predictions(df, xcols, ycol):
    assert xcols == ['utd', 'critic_params'], 'For now'
    model, slope, intercept, r2 = fit_log_linear(df, xcols, ycol)
    df[f'{ycol}_loglinear'] = predict_log_linear(df, xcols, slope, intercept)
    return df, (model, slope, intercept, r2)


def insert_predictions_interaction(df, xcols, ycol):
    assert xcols == ['utd', 'critic_params'], 'For now'
    model, slope, intercept, r2 = fit_log_linear_interaction(df, xcols, ycol)
    df[f'{ycol}_loglinear_interact'] = predict_log_linear_interaction(df, xcols, slope, intercept)
    return df, (model, slope, intercept, r2)


def insert_predictions_shared_slope(df, xcols, ycol):
    idx = ~df[ycol].isna()
    model, shared_slope, env_intercepts, r2 = fit_log_linear_shared_slope(df.loc[idx], xcols, ycol)
    df.loc[idx, f'{ycol}_loglinear_shared'] = predict_log_linear_shared_slope(
        df.loc[idx], xcols, shared_slope, env_intercepts
    )
    return df, (model, shared_slope, env_intercepts, r2)


def insert_predictions_shared_slope_interaction(df, xcols, ycol):
    model, shared_slope, env_intercepts, r2 = fit_log_linear_shared_slope_interaction(
        df, xcols, ycol
    )
    df[f'{ycol}_loglinear_shared_interact'] = predict_log_linear_shared_slope_interaction(
        df, xcols, shared_slope, env_intercepts
    )
    return df, (model, shared_slope, env_intercepts, r2)


def _load_params(name, description):
    params_path = os.path.join(fits_save_dir, f'{name}/{description}.npy')
    if os.path.exists(params_path):
        result = np.load(params_path, allow_pickle=True).item()
        print(f'Loaded params from {params_path}')
        return result
    else:
        return None


def _save_params(name, description, params):
    params_path = os.path.join(fits_save_dir, f'{name}/{description}.npy')
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    np.save(params_path, params, allow_pickle=True)
    print(f'Saved params to {params_path}')


def insert_predictions_powerlaw(df, xcols, ycol, name='', use_cached=True, parallel=True):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_powerlaw'

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    else:

        def helper(args):
            env, utd = args
            subset = df.query(f'env_name=="{env}" and utd=={utd}')
            if len(subset) == 0:
                return None
            x = subset['critic_params'].values
            y = subset[ycol].values
            a, b, c, b_unscaled, c_unscaled = fit_powerlaw(x, y, parallel=False)
            return (env, utd, a, b, c, b_unscaled, c_unscaled)

        tasks = [(env, utd) for env in df['env_name'].unique() for utd in df['utd'].unique()]

        # parallelize making fit per group; each fit is done serially
        num_threads = os.cpu_count() // 2
        if num_threads <= 1:
            parallel = False
        if parallel:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(tqdm(executor.map(helper, tasks)), total=len(tasks))
        else:
            results = [helper(task) for task in tasks]
        results = [r for r in results if r is not None]
        results = sorted(results, key=lambda x: (x[0], x[1]))
        params_dict = {(r[0], r[1]): r[2:] for r in results}
        _save_params(name, description, params_dict)

    df[f'{ycol}_powerlaw'] = predict_powerlaw(df, xcols, params_dict)
    return df, params_dict


def insert_predictions_powerlaw_shared_exponent(
    df, xcols, ycol, name='', use_cached=True, use_jax=False
):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_powerlaw_shared_exp'
    params_dict = None

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    if params_dict is None:
        group_data = {}
        for (env, utd), group in df.groupby(['env_name', 'utd']):
            if len(group) == 0:
                continue
            x = group['critic_params'].values
            y = group[ycol].values
            group_data[(env, utd)] = (x, y)
        params_dict = fit_powerlaw_shared_exponent(group_data, use_jax=use_jax)
        _save_params(name, description, params_dict)

    df[f'{ycol}_powerlaw_shared_exp'] = predict_powerlaw_shared_exponent(df, xcols, params_dict)
    return df, params_dict


def insert_predictions_inverse_power(df, xcols, ycol, name='', use_cached=True, parallel=True):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_inverse_power'
    params_dict = None

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    if params_dict is None:

        def helper(args):
            env, utd = args
            subset = df.query(f'env_name=="{env}" and utd=={utd}')
            if len(subset) == 0:
                return None
            x = subset['critic_params'].values
            y = subset[ycol].values
            a, b, c, a_unscaled, b_unscaled = fit_inverse_power(x, y, parallel=False)
            return (env, utd, a, b, c, a_unscaled, b_unscaled)

        tasks = [(env, utd) for env in df['env_name'].unique() for utd in df['utd'].unique()]

        # parallelize making fit per group; each fit is done serially
        num_threads = os.cpu_count() // 2
        if num_threads <= 1:
            parallel = False
        if parallel:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(tqdm(executor.map(helper, tasks)), total=len(tasks))
        else:
            results = [helper(task) for task in tasks]
        results = [r for r in results if r is not None]
        results = sorted(results, key=lambda x: (x[0], x[1]))
        params_dict = {(r[0], r[1]): r[2:] for r in results}
        _save_params(name, description, params_dict)

    df[f'{ycol}_inverse_power'] = predict_inverse_power(df, xcols, params_dict)
    return df, params_dict


def insert_predictions_inverse_power_shared_exponent(
    df, xcols, ycol, name='', loss_p=2, use_cached=True, use_jax=False
):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_inverse_power_shared_exp'
    if loss_p != 2:
        description += f'_p{loss_p}'
    params_dict = None

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    if params_dict is None:
        group_data = {}

        for (env, utd), group in df.groupby(['env_name', 'utd']):
            if len(group) == 0:
                continue
            x = group['critic_params'].values
            y = group[ycol].values
            group_data[(env, utd)] = (x, y)

        params_dict = fit_inverse_power_shared_exponent(group_data, loss_p=loss_p, use_jax=use_jax)
        _save_params(name, description, params_dict)

    df[f'{ycol}_inverse_power_shared_exp'] = predict_inverse_power_shared_exponent(
        df, xcols, params_dict
    )
    return df, params_dict


def _insert_predictions_bivariate(
    fit_type, fit_fn, predict_fn, df, xcols, ycol, name='', use_cached=True, **fit_kw
):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_{fit_type}'
    params_dict = None
    idx = (~df[ycol].isna()) & (df[ycol] < np.inf) & (df[ycol] > -np.inf)

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    if params_dict is None:
        params_dict = {}
        # init_params_per_env = fit_kw.pop('init_params_per_env', None)
        save_params = fit_kw.pop('save_params', True)
        for env, group in df.loc[idx].groupby('env_name'):
            if len(group) == 0:
                continue
            x = group[xcols].values
            x1, x2 = x[:, 0], x[:, 1]
            y = group[ycol].values
            # if init_params_per_env is not None:
            #     fit_kw['init_p'] = init_params_per_env[env]
            params_dict[env] = fit_fn(x1, x2, y, **fit_kw)
        if save_params:
            _save_params(name, description, params_dict)

    df.loc[idx, f'{ycol}_{fit_type}'] = predict_fn(df.loc[idx], xcols, params_dict)
    return df, params_dict


def insert_predictions_sum_powerlaw(df, xcols, ycol, name='', use_cached=True, **fit_kw):
    return _insert_predictions_bivariate(
        'sum_powerlaw',
        fit_sum_of_powerlaw,
        predict_sum_of_powerlaw,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_sum_powerlaw_shift(df, xcols, ycol, name='', use_cached=True, **fit_kw):
    return _insert_predictions_bivariate(
        'sum_powerlaw_shift',
        fit_sum_of_powerlaw_shift,
        predict_sum_of_powerlaw_shift,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_inverse_power_product(df, xcols, ycol, name='', use_cached=True, **fit_kw):
    return _insert_predictions_bivariate(
        'inverse_power_product',
        fit_inverse_power_product,
        predict_inverse_power_product,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_inverse_power_product_log_normalize(
    df, xcols, ycol, name='', use_cached=True, **fit_kw
):
    return _insert_predictions_bivariate(
        'inverse_power_product',
        fit_inverse_power_product_log_normalize,
        predict_inverse_power_product_log_normalize,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_inverse_power_flip_product(
    df, xcols, ycol, name='', use_cached=True, **fit_kw
):
    return _insert_predictions_bivariate(
        'inverse_power_flip_product',
        fit_inverse_power_flip_product,
        predict_inverse_power_flip_product,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_inverse_power_numerator_product(
    df, xcols, ycol, name='', use_cached=True, **fit_kw
):
    return _insert_predictions_bivariate(
        'inverse_power_numerator_product',
        fit_inverse_power_numerator_product,
        predict_inverse_power_numerator_product,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_inverse_power_numerator_flip_product(
    df, xcols, ycol, name='', use_cached=True, **fit_kw
):
    return _insert_predictions_bivariate(
        'inverse_power_numerator_flip_product',
        fit_inverse_power_numerator_flip_product,
        predict_inverse_power_numerator_flip_product,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_denominator_sum_power(df, xcols, ycol, name='', use_cached=True, **fit_kw):
    return _insert_predictions_bivariate(
        'denominator_sum_power',
        fit_denominator_sum_power,
        predict_denominator_sum_power,
        df,
        xcols,
        ycol,
        name,
        use_cached,
        **fit_kw,
    )


def insert_predictions_sum_power(df, xcols, ycol, name='', use_cached=True, **fit_kw):
    return _insert_predictions_bivariate(
        'sum_power', fit_sum_power, predict_sum_power, df, xcols, ycol, name, use_cached, **fit_kw
    )


def insert_predictions_sum_power(df, xcols, ycol, name='', use_cached=True, **fit_kw):
    _insert_predictions_bivariate(
        'sum_power', fit_sum_power, predict_sum_power, df, xcols, ycol, name, use_cached, **fit_kw
    )


def insert_predictions_sum_of_powerlaw_shared_exponent(
    df,
    xcols,
    ycol,
    name='',
    use_cached=True,
    use_jax=False,
    **fit_kw,
):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_sum_of_powerlaw_shared_exp'
    description += '' if fit_kw.get('log_loss', False) else '_no_log_loss'
    if fit_kw.get('loss_p', 2) != 2:
        description += f'_p{fit_kw.get("loss_p", 2)}'
    params_dict = None
    idx = ~df[ycol].isna()

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    if params_dict is None:
        group_data = {}
        save_params = fit_kw.pop('save_params', True)

        for env, group in df.loc[idx].groupby('env_name'):
            if len(group) == 0:
                continue
            x = group[xcols].values
            x1, x2 = x[:, 0], x[:, 1]
            y = group[ycol].values
            group_data[env] = (x1, x2, y)

        params_dict = fit_sum_of_powerlaw_shared_exponent(group_data, use_jax=use_jax, **fit_kw)
        if save_params:
            _save_params(name, description, params_dict)

    df.loc[idx, f'{ycol}_sum_of_powerlaw_shared_exp'] = predict_sum_of_powerlaw_shared_exponent(
        df.loc[idx], xcols, params_dict
    )
    return df, params_dict


def insert_predictions_sum_of_powerlaw_grouped(
    dfs: Dict[str, pd.DataFrame],
    groups,
    xcols,
    ycol,
    name='',
    use_cached=True,
    use_jax=False,
    **fit_kw,
):
    assert xcols == ['utd', 'critic_params'], 'For now'
    description = f'{xcols[0]}_{xcols[1]}_{ycol}_sum_of_powerlaw_grouped'
    description += '' if fit_kw.get('log_loss', False) else '_no_log_loss'
    if fit_kw.get('loss_p', 2) != 2:
        description += f'_p{fit_kw.get("loss_p", 2)}'
    params_dict = None
    idxs = [~df[ycol].isna() for df in dfs.values()]

    if use_cached:
        assert name
        params_dict = _load_params(name, description)

    if params_dict is None:
        envs = sorted(dfs[groups[0][0]]['env_name'].unique())
        params_dict = {}
        for group_labels in groups:
            group_dfs = {i: dfs[i].loc[idxs[i]] for i in group_labels}
            for env in envs:
                fit_data = {}
                for i, df in group_dfs.items():
                    subset_df = df.query(f'env_name=="{env}"')
                    x = subset_df[xcols].values
                    x1, x2 = x[:, 0], x[:, 1]
                    y = subset_df[ycol].values
                    fit_data[i] = (x1, x2, y)

                params = fit_sum_of_powerlaw_shared_exponent(fit_data, use_jax=use_jax, **fit_kw)
                for i, p in params.items():
                    params_dict[(env, i)] = p
        _save_params(name, description, params_dict)

    save_key = f'{ycol}_sum_of_powerlaw_grouped'
    if fit_kw.get('loss_p', 2) != 2:
        save_key += f'_p{fit_kw.get("loss_p", 2)}'

    for i, df in dfs.items():
        thresh_params = {env: params for (env, j), params in params_dict.items() if j == i}
        df.loc[idxs[i], save_key] = predict_sum_of_powerlaw_shared_exponent(
            df.loc[idxs[i]], xcols, thresh_params
        )

    return dfs, params_dict


def _plot_optimal_hparam_fit_per_env_helper(
    df,
    group_col,
    group_label_col,
    xcol,
    ycol,
    ycol_std,
    title,
    predict_fn_info_pairs,
    title_fn,
    xlabel_fn,
    data_label='Bootstrap optimal',
    **kw,
):
    envs = get_envs(df)
    all_xs = set(df[xcol].unique())
    all_group_vals = set(df[group_col].unique())

    if kw.get('interpolated_df') is not None:
        interpolated_df = kw['interpolated_df']
        all_xs |= set(interpolated_df[xcol].unique())
        all_group_vals |= set(interpolated_df[group_col].unique())
    else:
        interpolated_df = pd.DataFrame(columns=df.columns)

    if kw.get('extrapolated_df') is not None:
        extrapolated_df = kw['extrapolated_df']
        all_xs |= set(extrapolated_df[xcol].unique())
        all_group_vals |= set(extrapolated_df[group_col].unique())
    else:
        extrapolated_df = pd.DataFrame(columns=df.columns)

    if kw.get('compute_optimal_df') is not None:
        compute_optimal_df = kw['compute_optimal_df']
        all_xs |= set(compute_optimal_df[xcol].unique())
        all_group_vals |= set(compute_optimal_df[group_col].unique())
    else:
        compute_optimal_df = pd.DataFrame(columns=df.columns)

    all_xs = sorted(list(all_xs))
    all_group_vals = sorted(list(all_group_vals))
    x_smooth = np.logspace(np.log10(min(all_xs)), np.log10(max(all_xs)), 100)

    def set_yscale(ax):
        if kw.get('yscale') == 'log2':
            ax.set_yscale('log', base=2)
        elif kw.get('yscale') == 'log':
            ax.set_yscale('log')

    fig, axes = plt.subplots(
        len(envs),
        len(all_group_vals),
        figsize=(3 * len(all_group_vals), 2.5 * len(envs)),
        # sharey=True,
    )
    axes = np.array(axes).reshape(len(envs), len(all_group_vals))
    handles, labels = [], []

    xticks_kw = {}
    if 'xlabel_rotation' in kw:
        xticks_kw['rotation'] = kw['xlabel_rotation']
    if 'xlabel_fontsize' in kw:
        xticks_kw['fontsize'] = kw['xlabel_fontsize']
    hard_ymin = kw.get('hard_ymin', float('-inf'))
    hard_ymax = kw.get('hard_ymax', float('inf'))
    # xmin, xmax = float('inf'), float('-inf')
    # ymin, ymax = float('inf'), float('-inf')

    for i, env in enumerate(envs):
        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')
        env_df = df.query(f'env_name=="{env}"')
        interpolated_env_df = interpolated_df.query(f'env_name=="{env}"')
        extrapolated_env_df = extrapolated_df.query(f'env_name=="{env}"')
        compute_optimal_env_df = compute_optimal_df.query(f'env_name=="{env}"')

        for j, group_val in enumerate(all_group_vals):
            ax = axes[i, j]
            dta = env_df.query(f'{group_col}=={group_val}').sort_values(xcol)
            interpolated_dta = interpolated_env_df.query(f'{group_col}=={group_val}').sort_values(
                xcol
            )
            extrapolated_dta = extrapolated_env_df.query(f'{group_col}=={group_val}').sort_values(
                xcol
            )
            compute_optimal_dta = compute_optimal_env_df.query(
                f'{group_col}=={group_val}'
            ).sort_values(xcol)

            flag = False
            if len(dta) > 0:
                assert dta[group_label_col].nunique() == 1
                group_label_val = dta[group_label_col].iloc[0]
                flag = True
            if len(interpolated_dta) > 0:
                assert interpolated_dta[group_label_col].nunique() == 1
                group_label_val = interpolated_dta[group_label_col].iloc[0]
                flag = True
            if len(extrapolated_dta) > 0:
                assert extrapolated_dta[group_label_col].nunique() == 1
                group_label_val = extrapolated_dta[group_label_col].iloc[0]
                flag = True
            if len(compute_optimal_dta) > 0:
                assert compute_optimal_dta[group_label_col].nunique() == 1
                group_label_val = compute_optimal_dta[group_label_col].iloc[0]
                flag = True
            if not flag:
                continue

            prediction_input_df = pd.DataFrame(
                {'env_name': env, xcol: x_smooth, group_label_col: group_label_val}
            )

            if len(dta) > 0:
                ax.errorbar(
                    dta[xcol],
                    dta[ycol],
                    yerr=dta[ycol_std] if ycol_std else None,
                    fmt='o',
                    capsize=3,
                    label=data_label,
                    color='tab:blue',
                )
            if len(interpolated_dta) > 0:
                ax.errorbar(
                    interpolated_dta[xcol],
                    interpolated_dta[ycol],
                    yerr=interpolated_dta[ycol_std] if ycol_std else None,
                    fmt='s',
                    capsize=3,
                    color='teal',
                )
            if len(extrapolated_dta) > 0:
                ax.errorbar(
                    extrapolated_dta[xcol],
                    extrapolated_dta[ycol],
                    yerr=extrapolated_dta[ycol_std] if ycol_std else None,
                    fmt='X',
                    capsize=3,
                    color='deepskyblue',
                )
            if len(compute_optimal_dta) > 0:
                ax.errorbar(
                    compute_optimal_dta[xcol],
                    compute_optimal_dta[ycol],
                    yerr=compute_optimal_dta[ycol_std] if ycol_std else None,
                    fmt='*',
                    capsize=3,
                    color='gold',
                )

            for predict_fn, info in predict_fn_info_pairs:
                label = info['label']
                try:
                    fit_predictions = predict_fn(prediction_input_df)
                except Exception as e:
                    print('Exception:', e)
                    continue
                prediction_input_df[f'predictions_{label}'] = fit_predictions
                (line,) = ax.plot(x_smooth, fit_predictions, label=label, color=info.get('color'))
                ymin = min(ymin, max(min(fit_predictions), hard_ymin))
                ymax = max(ymax, min(max(fit_predictions), hard_ymin))

                if 'asymptote' in info:
                    color = line.get_color()
                    if (env, group_label_val) not in info['asymptote']:
                        continue
                    asymptote = info['asymptote'][(env, group_label_val)]
                    if asymptote > hard_ymin and asymptote < hard_ymax:
                        ax.axhline(y=asymptote, color=color, linestyle='--')
                        ymin = min(ymin, asymptote)
                        ymax = max(ymax, asymptote)

            if env != '':
                ax.set_title(f'{env}, {title_fn(group_label_val)}')
            else:
                ax.set_title(title_fn(group_label_val))
            xvals = sorted(
                list(
                    set(dta[xcol])
                    | set(interpolated_dta[xcol])
                    | set(extrapolated_dta[xcol])
                    | set(compute_optimal_dta[xcol])
                )
            )
            yvals = sorted(
                list(
                    set(dta[ycol])
                    | set(interpolated_dta[ycol])
                    | set(extrapolated_dta[ycol])
                    | set(compute_optimal_dta[ycol])
                )
            )

            ax.set_xlabel(xcol)
            ax.set_xscale('log', base=2)
            set_yscale(ax)
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xticks(xvals)
            ax.set_xticklabels([xlabel_fn(x) for x in xvals], **xticks_kw)

            if kw.get('yscale') == 'log2':
                ax.set_yscale('log', base=2)
            else:
                ax.set_yscale('log')
            ax.grid(alpha=0.3)

            xmin, xmax = min(xmin, min(xvals)), max(xmax, max(xvals))
            ymin, ymax = min(ymin, min(yvals)), max(ymax, max(yvals))
            handles, labels = ax.get_legend_handles_labels()

        for ax in axes[i]:
            ax.set_xlim(*expand_log_range(xmin, xmax))
            ax.set_ylim(*expand_log_range(ymin, ymax))

    # for ax in axes.flatten():
    #     ax.set_xlim(*expand_log_range(xmin, xmax))
    #     ax.set_ylim(*expand_log_range(ymin, ymax))

    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        loc='upper center',
        fontsize=14,
    )
    fig.suptitle(f'{title}, grouped by {group_label_col}', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_optimal_hparam_fit_per_env_n(
    df, ycol, ycol_std, title, predict_fn_info_pairs, group_col='critic_width', **kw
):
    _plot_optimal_hparam_fit_per_env_helper(
        df,
        group_col=group_col,
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


def plot_optimal_hparam_fit_per_env_utd(df, ycol, ycol_std, title, predict_fn_info_pairs, **kw):
    _plot_optimal_hparam_fit_per_env_helper(
        df,
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


def _plot_optimal_hparam_fit_per_env_helper_pretty(
    fig,
    axes,
    df,
    group_col,
    group_label_col,
    xcol,
    ycol,
    ycol_std,
    title,
    predict_fn_info_pairs,
    title_fn,
    xlabel_fn,
    data_label='Bootstrap optimal',
    legend=True,
    **kw,
):
    envs = get_envs(df)
    all_xs = set(df[xcol].unique())
    all_group_vals = set(df[group_col].unique())

    if kw.get('interpolated_df') is not None:
        interpolated_df = kw['interpolated_df']
        all_xs |= set(interpolated_df[xcol].unique())
        all_group_vals |= set(interpolated_df[group_col].unique())
    else:
        interpolated_df = pd.DataFrame(columns=df.columns)

    if kw.get('extrapolated_df') is not None:
        extrapolated_df = kw['extrapolated_df']
        all_xs |= set(extrapolated_df[xcol].unique())
        all_group_vals |= set(extrapolated_df[group_col].unique())
    else:
        extrapolated_df = pd.DataFrame(columns=df.columns)

    if kw.get('compute_optimal_df') is not None:
        compute_optimal_df = kw['compute_optimal_df']
        all_xs |= set(compute_optimal_df[xcol].unique())
        all_group_vals |= set(compute_optimal_df[group_col].unique())
    else:
        compute_optimal_df = pd.DataFrame(columns=df.columns)

    all_xs = sorted(list(all_xs))
    all_group_vals = sorted(list(all_group_vals))
    x_smooth = np.logspace(np.log10(min(all_xs)), np.log10(max(all_xs)), 100)

    def set_yscale(ax):
        if kw.get('yscale') == 'log2':
            ax.set_yscale('log', base=2)
        elif kw.get('yscale') == 'log':
            ax.set_yscale('log')

    axes = np.array(axes).reshape(len(envs), len(all_group_vals))

    xticks_kw = {}
    if 'xlabel_rotation' in kw:
        xticks_kw['rotation'] = kw['xlabel_rotation']
    if 'xlabel_fontsize' in kw:
        xticks_kw['fontsize'] = kw['xlabel_fontsize']
    hard_ymin = kw.get('hard_ymin', float('-inf'))
    hard_ymax = kw.get('hard_ymax', float('inf'))
    # xmin, xmax = float('inf'), float('-inf')
    # ymin, ymax = float('inf'), float('-inf')

    def get_qscaled_color(s):
        if not s.startswith('qscaled_'):
            return s
        else:
            return qscaled_plot_utils.COLORS[int(s[len('qscaled_') :])]

    for i, env in enumerate(envs):
        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')
        env_df = df.query(f'env_name=="{env}"')
        interpolated_env_df = interpolated_df.query(f'env_name=="{env}"')
        extrapolated_env_df = extrapolated_df.query(f'env_name=="{env}"')
        compute_optimal_env_df = compute_optimal_df.query(f'env_name=="{env}"')

        for j, group_val in enumerate(all_group_vals):
            ax = axes[i, j]
            dta = env_df.query(f'{group_col}=={group_val}').sort_values(xcol)
            interpolated_dta = interpolated_env_df.query(f'{group_col}=={group_val}').sort_values(
                xcol
            )
            extrapolated_dta = extrapolated_env_df.query(f'{group_col}=={group_val}').sort_values(
                xcol
            )
            compute_optimal_dta = compute_optimal_env_df.query(
                f'{group_col}=={group_val}'
            ).sort_values(xcol)

            flag = False
            if len(dta) > 0:
                assert dta[group_label_col].nunique() == 1
                group_label_val = dta[group_label_col].iloc[0]
                flag = True
            if len(interpolated_dta) > 0:
                assert interpolated_dta[group_label_col].nunique() == 1
                group_label_val = interpolated_dta[group_label_col].iloc[0]
                flag = True
            if len(extrapolated_dta) > 0:
                assert extrapolated_dta[group_label_col].nunique() == 1
                group_label_val = extrapolated_dta[group_label_col].iloc[0]
                flag = True
            if len(compute_optimal_dta) > 0:
                assert compute_optimal_dta[group_label_col].nunique() == 1
                group_label_val = compute_optimal_dta[group_label_col].iloc[0]
                flag = True
            if not flag:
                continue

            prediction_input_df = pd.DataFrame(
                {'env_name': env, xcol: x_smooth, group_label_col: group_label_val}
            )

            if len(dta) > 0:
                ax.errorbar(
                    dta[xcol],
                    dta[ycol],
                    # yerr=dta[ycol_std] if ycol_std else None,
                    fmt='o',
                    capsize=3,
                    label=data_label,
                    color=get_qscaled_color('qscaled_1'),
                    markersize=10,
                    alpha=0.8,
                    # color='tab:blue',
                )
            if len(interpolated_dta) > 0:
                ax.errorbar(
                    interpolated_dta[xcol],
                    interpolated_dta[ycol],
                    yerr=interpolated_dta[ycol_std] if ycol_std else None,
                    fmt='s',
                    capsize=3,
                    color='teal',
                )
            if len(extrapolated_dta) > 0:
                ax.errorbar(
                    extrapolated_dta[xcol],
                    extrapolated_dta[ycol],
                    yerr=extrapolated_dta[ycol_std] if ycol_std else None,
                    fmt='X',
                    capsize=3,
                    color='deepskyblue',
                )
            if len(compute_optimal_dta) > 0:
                ax.errorbar(
                    compute_optimal_dta[xcol],
                    compute_optimal_dta[ycol],
                    yerr=compute_optimal_dta[ycol_std] if ycol_std else None,
                    fmt='*',
                    capsize=3,
                    color='gold',
                )

            for predict_fn, info in predict_fn_info_pairs:
                label = info['label']
                try:
                    fit_predictions = predict_fn(prediction_input_df)
                except Exception as e:
                    print('Exception:', e)
                    continue
                prediction_input_df[f'predictions_{label}'] = fit_predictions
                # (line,) = ax.plot(x_smooth, fit_predictions, label=label, color=info.get('color'))
                (line,) = ax.plot(
                    x_smooth,
                    fit_predictions,
                    label=label,
                    color=get_qscaled_color(info.get('color')),
                    linewidth=3,
                )
                ymin = min(ymin, max(min(fit_predictions), hard_ymin))
                ymax = max(ymax, min(max(fit_predictions), hard_ymin))

                # point_predictions = predict_fn(pd.DataFrame({'env_name': env, xcol: dta[xcol], }))

                # if 'asymptote' in info:
                if False:
                    # color = line.get_color()
                    if (env, group_label_val) not in info['asymptote']:
                        continue
                    asymptote = info['asymptote'][(env, group_label_val)]
                    if asymptote > hard_ymin and asymptote < hard_ymax:
                        ax.axhline(y=asymptote, color=color, linestyle='--')
                        ymin = min(ymin, asymptote)
                        ymax = max(ymax, asymptote)

            if env != '':
                ax.set_title(f'{env}, {title_fn(group_label_val)}', fontsize='xx-large')
            else:
                ax.set_title(title_fn(group_label_val), fontsize='xx-large')
            xvals = sorted(
                list(
                    set(dta[xcol])
                    | set(interpolated_dta[xcol])
                    | set(extrapolated_dta[xcol])
                    | set(compute_optimal_dta[xcol])
                )
            )
            yvals = sorted(
                list(
                    set(dta[ycol])
                    | set(interpolated_dta[ycol])
                    | set(extrapolated_dta[ycol])
                    | set(compute_optimal_dta[ycol])
                )
            )

            # ax.set_xlabel(xcol)
            ax.set_xscale('log', base=2)
            # set_yscale(ax)

            if kw.get('yscale') == 'log2':
                ax.set_yscale('log', base=2)
            else:
                ax.set_yscale('log')

            if i == len(envs) - 1 or kw.get('inner_axis_label', True):
                if xcol == 'critic_params':
                    xlabel = r'$N$: model size'
                elif xcol == 'utd':
                    xlabel = r'$\sigma$: UTD'
                else:
                    raise ValueError(f'Unknown xcol: {xcol}')
            else:
                xlabel = ''

            if ycol == 'best_bs_bootstrap_mean':
                ylabel = r'$B^*$: Best batch size' if j == 0 else ''
            else:
                raise ValueError

            rliable_plot_utils._annotate_and_decorate_axis(
                ax,
                xlabel=xlabel,
                ylabel=ylabel,
                labelsize='xx-large',
                ticklabelsize='xx-large',
                grid_alpha=0.2,
                legend=False,
            )

            # qscaled_plot_utils.ax_set_x_bounds_and_scale(ax, xticks=xvals)

            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xticks(xvals)
            ax.set_xticklabels([xlabel_fn(x) for x in xvals])

            qscaled_plot_utils.ax_set_y_bounds_and_scale(ax, yticks=yvals)

            # ax.grid(alpha=0.3)

            xmin, xmax = min(xmin, min(xvals)), max(xmax, max(xvals))
            ymin, ymax = min(ymin, min(yvals)), max(ymax, max(yvals))
            handles, labels = ax.get_legend_handles_labels()

        for ax in axes[i]:
            ax.set_xlim(*expand_log_range(xmin, xmax))
            if kw.get('set_ylim', True):
                ax.set_ylim(*expand_log_range(ymin, ymax))

    # for ax in axes.flatten():
    #     ax.set_xlim(*expand_log_range(xmin, xmax))
    #     ax.set_ylim(*expand_log_range(ymin, ymax))

    if legend:
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 0),
            ncol=2,
            loc='upper center',
            fontsize=14,
        )
    # fig.suptitle(f'{title}, grouped by {group_label_col}', y=1.02, fontsize=16)
    # plt.tight_layout()
    # plt.show()


def _plot_optimal_hparam_fit_per_env_n_pretty(
    fig, axes, df, ycol, ycol_std, title, predict_fn_info_pairs, group_col='critic_width', **kw
):
    _plot_optimal_hparam_fit_per_env_helper_pretty(
        fig,
        axes,
        df,
        group_col=group_col,
        group_label_col='critic_params',
        xcol='utd',
        ycol=ycol,
        ycol_std=ycol_std,
        title=title,
        predict_fn_info_pairs=predict_fn_info_pairs,
        title_fn=lambda x: f'$N$={abbreviate_number(x)}',
        xlabel_fn=lambda x: int(x),
        legend=False,
        data_label='Empirical value',
        **kw,
    )


def _plot_optimal_hparam_fit_per_env_utd_pretty(
    fig, axes, df, ycol, ycol_std, title, predict_fn_info_pairs, **kw
):
    _plot_optimal_hparam_fit_per_env_helper_pretty(
        fig,
        axes,
        df,
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
        legend=False,
        data_label='Empirical value',
        **kw,
    )


def plot_optimal_hparam_fit_per_env_n_pretty(
    df, ycol, ycol_std, title, predict_fn_info_pairs, group_col='critic_width', **kw
):
    envs = get_envs(df)
    fig, axes = plt.subplots(
        len(envs),
        len(df[group_col].unique()),
        figsize=(4 * len(df[group_col].unique()), 3 * len(envs)),
    )
    axes = np.array(axes).reshape(len(envs), len(df[group_col].unique()))
    _plot_optimal_hparam_fit_per_env_n_pretty(
        fig, axes, df, ycol, ycol_std, title, predict_fn_info_pairs, group_col='critic_width', **kw
    )
    yticks = [32, 128, 512, 2048]
    for ax in axes.flatten():
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks])
        ax.set_ylim(expand_log_range(min(yticks), max(yticks)))
    plt.tight_layout()
    if 'save_path' in kw:
        os.makedirs(os.path.dirname(kw['save_path']), exist_ok=True)
        plt.savefig(kw['save_path'])
    plt.show()


def plot_optimal_hparam_fit_per_env_utd_pretty(
    df, ycol, ycol_std, title, predict_fn_info_pairs, **kw
):
    envs = get_envs(df)
    fig, axes = plt.subplots(
        len(envs),
        len(df['utd'].unique()),
        figsize=(4 * len(df['utd'].unique()), 3 * len(envs)),
    )
    axes = np.array(axes).reshape(len(envs), len(df['utd'].unique()))
    _plot_optimal_hparam_fit_per_env_utd_pretty(
        fig, axes, df, ycol, ycol_std, title, predict_fn_info_pairs, **kw
    )
    yticks = [32, 128, 512, 2048]
    for ax in axes.flatten():
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks])
        ax.set_ylim(expand_log_range(min(yticks), max(yticks)))
    plt.tight_layout()
    if 'save_path' in kw:
        os.makedirs(os.path.dirname(kw['save_path']), exist_ok=True)
        plt.savefig(kw['save_path'])
    plt.show()


def plot_optimal_hparam_fit_per_env_combined_pretty(
    df, ycol, ycol_std, title, predict_fn_info_pairs, **kw
):
    qscaled_plot_utils.set_theme()
    envs = get_envs(df)
    assert len(envs) == 1
    x = len(df['utd'].unique())
    x_ = len(df['critic_params'].unique())
    assert x == x_
    fig, axes = plt.subplots(2, x, figsize=(4.25 * x, 3.25 * 2))
    axes = np.array(axes).reshape(2, x)
    _plot_optimal_hparam_fit_per_env_utd_pretty(
        fig, axes[0], df, ycol, ycol_std, title, predict_fn_info_pairs, **kw
    )
    _plot_optimal_hparam_fit_per_env_n_pretty(
        fig, axes[1], df, ycol, ycol_std, title, predict_fn_info_pairs, **kw
    )
    yticks = [32, 128, 512, 2048]
    for ax in axes.flatten():
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks])

    handles, labels = axes[-1, -1].get_legend_handles_labels()
    handles = [handles[-1]] + handles[:-1]
    labels = [labels[-1]] + labels[:-1]
    axes[-1, -1].legend(handles, labels, prop={'size': 14}, ncol=1)

    plt.tight_layout(pad=0.0, h_pad=2.0)  # Add vertical padding
    if 'save_path' in kw:
        os.makedirs(os.path.dirname(kw['save_path']), exist_ok=True)
        plt.savefig(kw['save_path'])
    plt.show()
