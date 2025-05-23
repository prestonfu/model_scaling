import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import statsmodels.api as sm

from scripts.utils import abbreviate_number, clean_sci


def huhhh(df):
    print(df.iloc[0]['mean_sep_new_data_overfitting'])


def plot_metric_over_training_3d_multiple_metrics(
    df,
    metrics,
    suptitle,
    row_group_name,
    col_group_name,
    row_label_name=None,
    col_label_name=None,
    title_fn=None,
    **kw,
):
    """
    Adapted from `_plot_metric_over_training_3d`, but for multiple metrics,
    but only one configuration per plot.
    """
    envs = sorted(df['env_name'].unique())
    assert len(envs) == 1
    env = envs[0]
    row_group_vals = sorted(df[row_group_name].unique())
    col_group_vals = sorted(df[col_group_name].unique())
    n_rows = len(row_group_vals)
    n_cols = len(col_group_vals)

    if row_label_name is None:
        row_label_name = row_group_name
    if col_label_name is None:
        col_label_name = col_group_name

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    axes = np.array(axes).reshape(len(row_group_vals), len(col_group_vals))
    lines, labels = [], []

    ymin, ymax = float('inf'), float('-inf')
    xmin, xmax = float('-inf'), float('inf')
    if kw.get('xlim'):
        xmin, xmax = kw['xlim']

    for i, row_group_val in enumerate(row_group_vals):
        for j, col_group_val in enumerate(col_group_vals):
            subset = df.query(
                f'{row_group_name}=={row_group_val} and {col_group_name}=={col_group_val}'
            )
            if subset.empty:
                continue

            assert len(subset) == 1
            row_data = subset.iloc[0]
            row_label_val = row_data[row_label_name]
            col_label_val = row_data[col_label_name]

            ax = axes[i, j]
            if title_fn is not None:
                ax.set_title(title_fn(row_label_val, col_label_val))
            else:
                ax.set_title(f'{row_label_name}={row_label_val}, {col_label_name}={col_label_val}')
            ax.set_xlabel('env step')
            ax.grid(True, alpha=0.3)

            for metric in metrics:
                data = pd.DataFrame(
                    {
                        'training_step': row_data['training_step'],
                        f'mean_{metric}': row_data[f'mean_{metric}'],
                        f'std_{metric}': row_data[f'std_{metric}'],
                    }
                )
                data = data.dropna()
                data = data.rolling(5).mean().dropna()

                mask = (data['training_step'] >= xmin) & (data['training_step'] <= xmax)
                data = data[mask]

                (line,) = ax.plot(data['training_step'], data[f'mean_{metric}'])
                ax.fill_between(
                    data['training_step'],
                    data[f'mean_{metric}'] - data[f'std_{metric}'],
                    data[f'mean_{metric}'] + data[f'std_{metric}'],
                    alpha=0.2,
                    color=line.get_color(),
                )

                if metric not in labels:
                    labels.append(metric)
                    lines.append(line)

                this_ymin = (data[f'mean_{metric}'] - data[f'std_{metric}']).min()
                this_ymax = (data[f'mean_{metric}'] + data[f'std_{metric}']).max()
                # ymin = min(this_ymin, ymin, metric_config.get('hline', float('inf')))
                # ymax = max(this_ymax, ymax, metric_config.get('hline', float('-inf')))
                ymin = min(this_ymin, ymin)
                ymax = max(this_ymax, ymax)
                ymin = max(ymin, kw.get('hard_ymin', float('-inf')))
                ymax = min(ymax, kw.get('hard_ymax', float('inf')))

    for ax in axes.flatten():
        ax.set_xlim(xmin, xmax)
        if kw.get('yscale') is None:
            margin = 0.1 * (ymax - ymin)
            try:
                ax.set_ylim(ymin - margin, ymax + margin)
            except ValueError:
                fig.delaxes(ax)
        elif kw.get('yscale') in ['log', 'log2']:
            ymin = max(ymin, 1e-3)
            margin = 0.1 * (np.log10(ymax) - np.log10(ymin))
            try:
                ax.set_ylim(
                    10 ** (np.log10(ymin) - margin),
                    10 ** (np.log10(ymax) + margin),
                )
                if kw.get('yscale') == 'log2':
                    ax.set_yscale('log', base=2)
                else:
                    ax.set_yscale('log')
            except ValueError:
                fig.delaxes(ax)
        else:
            raise NotImplementedError

    fig.legend(
        lines,
        labels,
        loc='upper center',
        ncol=len(labels),
        fontsize=12,
        bbox_to_anchor=(0.5, 0),
    )
    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=16, y=1.04)
    plt.show()
