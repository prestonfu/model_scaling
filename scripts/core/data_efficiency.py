import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.linear_model import LinearRegression

from scripts.utils import abbreviate_number, expand_log_range
from scripts.core.fitting import fit_powerlaw


def fit_log_linear(df, xcol, ycol):
    x = np.log(df[xcol].values).reshape(-1, 1)
    y = np.log(df[ycol].values)
    model = LinearRegression().fit(x, y)
    return model.coef_[0], model.intercept_, model.score(x, y)


def fit_time_vs_critic_params_log(df, metric='time_to_threshold'):
    fits = {}
    groups = df.groupby(['env_name', 'utd'])
    for (env, utd), group_df in groups:
        if len(group_df) >= 2:
            fits[(env, utd)] = fit_log_linear(group_df, 'critic_params', metric)
    return fits


def fit_time_vs_utd_log(df, metric='time_to_threshold'):
    fits = {}
    groups = df.groupby(['env_name', 'critic_params'])
    for (env, width), group_df in groups:
        if len(group_df) >= 2:
            fits[(env, width)] = fit_log_linear(group_df, 'utd', metric)
    return fits


def compute_data_efficiency_per_env(df, envs):
    """Compute the data efficiency dictionary for each environment."""
    data_efficiency_dict = {}

    for env in envs:
        env_df = df[df['env_name'] == env]
        utds = sorted(env_df['utd'].unique())
        critic_params = sorted(env_df['critic_params'].unique())
        times = []
        for utd in utds:
            for critic_params_ in critic_params:
                subset = env_df[
                    (env_df['utd'] == utd) & (env_df['critic_params'] == critic_params_)
                ]
                if len(subset) == 0:
                    continue
                assert len(subset) == 1
                time_to_threshold = subset['time_to_threshold'].values[0][:]
                times.append((utd, critic_params_, time_to_threshold))

    return data_efficiency_dict


def plot_data_efficiency_per_env(df, n_thresholds, suptitle):
    """Multiple thresholds"""
    envs = sorted(df['env_name'].unique())

    def helper(group_col, group_label_col, xcol, title_fn, xlabel_fn, **kw):
        group_vals = sorted(df[group_col].unique())

        fig, axes = plt.subplots(
            len(envs),
            len(group_vals),
            figsize=(len(group_vals) * 2.5, len(envs) * 2.2),
            sharey='row',
        )
        fig.suptitle(
            f'Data Efficiency grouped by {group_label_col}' + suptitle, fontsize=14, y=1.02
        )
        xmin, xmax = expand_log_range(df[xcol].min(), df[xcol].max())
        colors = sns.color_palette('viridis', n_colors=n_thresholds)
        handles, labels = None, None

        for i, env in enumerate(envs):
            for j, group_val in enumerate(group_vals):
                subset = df[(df['env_name'] == env) & (df[group_col] == group_val)]
                if len(subset) == 0:
                    continue
                group_label_vals = subset[group_label_col].unique()
                assert len(group_label_vals) == 1
                group_label_val = group_label_vals[0]
                subset.sort_values(by=xcol, inplace=True)
                xs = subset[xcol].values
                thresholds = np.stack(subset['thresholds'].values)
                times_means = np.stack(subset['crossings'].values)
                times_stds = np.stack(subset['crossings_std'].values)
                for k in range(n_thresholds):
                    axes[i, j].plot(
                        xs, times_means[:, k], 'o-', label=f'Threshold {k + 1}', color=colors[k]
                    )
                    axes[i, j].fill_between(
                        xs,
                        times_means[:, k] - times_stds[:, k],
                        times_means[:, k] + times_stds[:, k],
                        alpha=0.2,
                        color=colors[k],
                    )

                ax = axes[i, j]
                ax.set_xlabel(xcol)
                if j == 0:
                    ax.set_ylabel('env steps to threshold')
                ax.set_title(f'{env}, {title_fn(group_label_val)}')
                ax.set_xscale('log')
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_xlim(xmin, xmax)
                ax.set_xticks(xs)
                if 'xtick_rotation' in kw:
                    ax.set_xticklabels([xlabel_fn(x) for x in xs], rotation=kw['xtick_rotation'])
                else:
                    ax.set_xticklabels([xlabel_fn(x) for x in xs])
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)

                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            ncol=int(np.ceil(n_thresholds / 2)),
        )
        plt.tight_layout()
        plt.show()

    helper(
        'critic_width',
        'critic_params',
        'utd',
        title_fn=lambda x: f'$N$={abbreviate_number(x)}',
        xlabel_fn=lambda x: int(x),
    )
    helper(
        'utd',
        'utd',
        'critic_params',
        title_fn=lambda x: f'$\sigma$={x}',
        xlabel_fn=abbreviate_number,
        xtick_rotation=90,
    )


def plot_data_efficiency_comparison(df_label_pairs, n_thresholds):
    """
    Compare data efficiency across methods, one figure per env.

    - df_label_pairs: list of (df, label) tuples
    - group_col: which column to iterate over for columns (e.g., 'critic_width')
    - group_label_col: label for grouping in the title (e.g., 'critic_params')
    - xcol: which column to plot on the x-axis
    - title_fn: function for per-column title
    - xlabel_fn: function to label x-axis ticks
    """

    def helper(group_col, group_label_col, xcol, title_fn, xlabel_fn, **kw):
        envs = sorted(set().union(*(df['env_name'].unique() for df, _ in df_label_pairs)))
        group_vals = sorted(set().union(*(df[group_col].unique() for df, _ in df_label_pairs)))
        all_xs = sorted(set().union(*(df[xcol].unique() for df, _ in df_label_pairs)))
        xmin, xmax = expand_log_range(min(all_xs), max(all_xs))

        colors = sns.color_palette('viridis', n_colors=n_thresholds)

        for env in envs:
            fig, axes = plt.subplots(
                len(df_label_pairs),
                len(group_vals),
                figsize=(len(group_vals) * 2.5, len(df_label_pairs) * 2.4),
                sharey=True,
            )
            axes = np.array(axes).reshape(len(df_label_pairs), len(group_vals))
            fig.suptitle(f'Data Efficiency grouped by {group_label_col} for {env}', fontsize=16)

            handles, labels = None, None

            ymin, ymax = np.inf, -np.inf

            for i, (df, label) in enumerate(df_label_pairs):
                for j, group_val in enumerate(group_vals):
                    subset = df[(df['env_name'] == env) & (df[group_col] == group_val)]
                    if len(subset) == 0:
                        continue
                    group_label_vals = subset[group_label_col].unique()
                    assert len(group_label_vals) == 1
                    group_label_val = group_label_vals[0]
                    subset = subset.sort_values(by=xcol)
                    xs = subset[xcol].values
                    thresholds = np.stack(subset['thresholds'].values)
                    times_means = np.stack(subset['crossings'].values)
                    times_stds = np.stack(subset['crossings_std'].values)

                    ax = axes[i, j]

                    for k in range(n_thresholds):
                        ax.plot(
                            xs, times_means[:, k], 'o-', label=f'Threshold {k + 1}', color=colors[k]
                        )
                        ax.fill_between(
                            xs,
                            times_means[:, k] - times_stds[:, k],
                            times_means[:, k] + times_stds[:, k],
                            alpha=0.2,
                            color=colors[k],
                        )

                    ymin = min(ymin, times_means.min())
                    ymax = max(ymax, times_means.max())

                    ax.set_title(f'{env}, {title_fn(group_label_val)}')
                    ax.set_xscale('log')
                    ax.xaxis.set_minor_locator(ticker.NullLocator())
                    ax.set_xlim(xmin, xmax)
                    ax.set_xticks(xs)
                    if 'xtick_rotation' in kw:
                        ax.set_xticklabels(
                            [xlabel_fn(x) for x in xs], rotation=kw['xtick_rotation']
                        )
                    else:
                        ax.set_xticklabels([xlabel_fn(x) for x in xs])
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)

                    if i == len(df_label_pairs) - 1:
                        ax.set_xlabel(xcol)
                    if j == 0:
                        ax.set_ylabel('env steps to threshold')

                    if handles is None:
                        handles, labels = ax.get_legend_handles_labels()

                # Add row label to the left of the row
                fig.text(
                    0.02,
                    1 - ((i + 0.5) / len(df_label_pairs)),
                    label,
                    va='center',
                    ha='right',
                    fontsize=14,
                    rotation=90,
                )

            ymin, ymax = expand_log_range(ymin, ymax)
            for ax in axes.flatten():
                ax.set_ylim(ymin, ymax)

            fig.legend(
                handles,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0),
                ncol=int(np.ceil(n_thresholds / 2)),
            )
            plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
            plt.show()

    helper(
        'critic_width',
        'critic_params',
        'utd',
        title_fn=lambda x: f'$N$={abbreviate_number(x)}',
        xlabel_fn=lambda x: int(x),
    )
    helper(
        'utd',
        'utd',
        'critic_params',
        title_fn=lambda x: f'$\sigma$={x}',
        xlabel_fn=abbreviate_number,
        xtick_rotation=90,
    )


def plot_data_efficiency_fits(
    df, groupby_cols, xcol, ycol, title_fmt, fits_dict, row_vals, col_vals
):
    """One threshold"""
    row_keys = sorted(df[row_vals].unique())
    col_keys = sorted(df[col_vals].unique())
    x_vals = sorted(df[xcol].unique())
    if xcol == 'critic_params':
        xlabels = [abbreviate_number(x) for x in x_vals]
    elif xcol == 'utd':
        xlabels = list(map(str, x_vals))
    else:
        raise NotImplementedError

    nrows = len(row_keys)
    ncols = len(col_keys)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.5 * ncols, 3 * nrows),
        sharex='row',
        sharey='row',
    )
    axes = axes.reshape(nrows, ncols)

    powerlaw_params = {}

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            ax = axes[i][j]

            if col_vals == 'critic_params':
                col_val_str = abbreviate_number(col_val)
            else:
                col_val_str = str(col_val)

            # Construct the group key
            if groupby_cols == ['env_name', 'utd']:
                key = (row_val, col_val)
            elif groupby_cols == ['env_name', 'critic_params']:
                key = (row_val, col_val)
            else:
                raise ValueError('Unsupported groupby_cols combination')

            group_df = df[(df[row_vals] == row_val) & (df[col_vals] == col_val)]
            group_df = group_df.sort_values(by=xcol)

            if key in fits_dict and len(group_df) >= 2:
                slope, intercept, r2 = fits_dict[key]
                r2_str = f'{r2:.2f}'
                x_vals = group_df[xcol].values
                y_vals = group_df[ycol].values
                x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_fit_loglinear = np.exp(slope * np.log(x_fit) + intercept)

                a, b, c = fit_powerlaw(x_vals, y_vals)
                x_fit_log = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 100)
                y_fit_powerlaw = powerlaw_fn(x_fit_log, a, b, c)
                powerlaw_formula = f'y={c:.2e} + (x / {b:.2e})^({-a:.2f})'
                powerlaw_params[col_val] = (a, b, c)

                ax.scatter(x_vals, y_vals, label='Data')
                ax.plot(
                    x_fit,
                    y_fit_loglinear,
                    color='gray',
                    linestyle='--',
                    label=f'y={np.exp(intercept):.2e} * x^{slope:.2f}',
                )
                ax.plot(
                    x_fit_log,
                    y_fit_powerlaw,
                    color='lightblue',
                    label=f'Asymptote {c:.2e}',
                )
                ax.axhline(c, color='lightblue', linestyle='-.')
                ax.set_title(
                    title_fmt.format(row_val, col_val_str) + r' ($R^2$=' + r2_str + ')',
                    fontsize=11,
                )
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_xticks(x_vals)
                ax.set_xticklabels(xlabels)
                ax.legend()
            else:
                ax.set_visible(False)

    fig.tight_layout()
    plt.suptitle(ycol, fontsize=14, y=1.04)
    plt.show()

    return powerlaw_params
