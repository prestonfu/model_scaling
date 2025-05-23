# Compute-Optimal Scaling for Value-Based Deep RL

Codebase based on [BRO](https://github.com/naumix/BiggerRegularizedOptimistic).

## Setup

Follow the installation steps for [SimbaV2](https://github.com/dojeon-ai/SimbaV2/tree/master). 
Note that you may need to install Jax per those installation steps
*after* installing `humanoid_bench`.

Then:
```bash
cd single && pip install -e .`
cd /path/to/qscaled-private && pip install -e .
```

## Running BRO algorithm

`python3 train_parallel.py --benchmark=dmc --env_name=dog-run --num_seeds=10 --updates_per_step=10`


## Reproducing results in the paper

* Figure 1: `single/scripts/experiments/utd_x_width_x_bs/overfitting.ipynb`
* Figure 2: `single/scripts/experiments/targets/side_critic.ipynb`
* Figures 3 onward: `single/scripts/experiments/utd_x_width_x_bs/main.ipynb`
* Averaged environment logic for DMC-medium and DMC-hard: `single/scripts/experiments/bro_ablations/dmc.ipynb`
