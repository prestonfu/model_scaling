import numpy as np
import pandas as pd
from datetime import datetime


def get_time_str():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def round_sigfigs(x, sig=3):
    return np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)


def abbreviate_number(num):
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f'{num / 1_000_000_000:.0f}B'
    elif abs_num >= 1_000_000:
        return f'{num / 1_000_000:.0f}M'
    elif abs_num >= 1_000:
        return f'{num / 1_000:.0f}K'
    else:
        return str(num)


def pretty_print_dict(d):
    max_length = max(len(key) for key in d.keys())
    for key, value in d.items():
        print(f'{key.ljust(max_length)}   {value}')


def tabulate_optimal_params(best_bs, best_lr):
    group_keys = ['env_name', 'utd', 'critic_width', 'critic_params']

    # only implemented this case for now
    assert best_bs.groupby(group_keys)['env_name'].count().max() == 1
    assert best_lr.groupby(group_keys)['env_name'].count().max() == 1

    best_bs = best_bs[group_keys + ['best_bs_bootstrap_mean']]
    best_lr = best_lr[group_keys + ['best_lr_bootstrap_mean']]

    df = pd.merge(best_bs, best_lr, on=group_keys)
    df['utd'] = df['utd'].astype(int)
    df['best_bs_bootstrap_mean'] = np.round(df['best_bs_bootstrap_mean']).astype(int)
    df['best_bs_bootstrap_mean_rounded'] = (
        np.round(df['best_bs_bootstrap_mean'] / 16).astype(int) * 16
    )
    df['best_lr_bootstrap_mean'] = df['best_lr_bootstrap_mean'].apply(lambda x: round_sigfigs(x, 3))

    return df


def expand_range(lower, upper, amount=0.1):
    return (1 + amount) * lower - amount * upper, (1 + amount) * upper - amount * lower


def expand_log_range(lower, upper, amount=0.1):
    lower, upper = expand_range(np.log(lower), np.log(upper), amount)
    return np.exp(lower), np.exp(upper)


def clean_sci(x):
    return f'{x:.2e}'.replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')


def drop_outliers_iqr_series(series):
    series = pd.Series(series)
    Q1 = series.dropna().quantile(0.25)
    Q3 = series.dropna().quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series.isna()) | ((series >= lower) & (series <= upper))]


def drop_outliers_iqr_df(df, column):
    Q1 = df[column].dropna().quantile(0.25)
    Q3 = df[column].dropna().quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column].isna()) | ((df[column] >= lower) & (df[column] <= upper))]
