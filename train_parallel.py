import os

os.environ['MUJOCO_GL'] = 'egl'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if 'TPU_VISIBLE_DEVICES' not in os.environ:
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    tpu_id = None
    os.environ['EGL_DEVICE_ID'] = gpu_id
    os.environ['MUJOCO_EGL_DEVICE_ID'] = gpu_id
else:
    tpu_id = os.environ.get('TPU_VISIBLE_DEVICES')
    gpu_id = None
    os.environ['EGL_DEVICE_ID'] = '0'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

import random
import time
from copy import deepcopy

import numpy as np
import wandb
from absl import app, flags
from jaxrl.agents import agents
from jaxrl.envs.new_env import make_env
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import (
    RollingMeter,
    Timer,
    init_wandb,
    is_using_slurm,
    log_seeds_to_wandb,
    mute_warning,
    prefix_metrics,
)
from ml_collections import config_flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', 'tmp', 'Directory to save data to.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('max_steps', int(2000000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(2500), 'Number of training steps to start training.')
flags.DEFINE_integer('num_seeds', 5, 'Number of parallel seeds to run.')
flags.DEFINE_string('benchmark', 'dmc', 'Environment name.')
flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('offline_evaluation', True, 'Do offline eval.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm.')
flags.DEFINE_string('wandb_entity', 'anonymous', 'Wandb entity.')
flags.DEFINE_string('wandb_project', 'BRO_model_scaling', 'Wandb project.')
flags.DEFINE_string('wandb_tag', None, 'Wandb tag.')

flags.DEFINE_boolean('validation', True, 'Whether to log validation.')
flags.DEFINE_integer(
    'validation_interval', 10, 'Validation frequency (relative to train env step).'
)

flags.DEFINE_boolean(
    'log_replay_tderr',
    True,
    'Whether to log td error on data sampled from replay buffer.',
)
flags.DEFINE_integer('replay_compute_interval', 500, 'New data td error computation frequency.')
flags.DEFINE_integer('replay_oldnew_batch_size', 256, 'Batch size for new data.')

flags.DEFINE_boolean('log_old_target_tderr', True, 'Whether to log td error on old target.')
flags.DEFINE_integer(
    'old_target_compute_interval',
    1000,
    'Old target network TD error computation frequency (env steps).',
)
flags.DEFINE_list(
    'old_target_update_intervals',
    ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512'],
    'Old target network update frequency (env steps).',
)

flags.DEFINE_boolean('log_grad_var', True, 'Whether to log td error on old target.')
flags.DEFINE_integer('grad_var_compute_interval', 500, 'Gradient variance computation frequency.')
flags.DEFINE_integer('grad_var_batch_size', 256, 'Batch size for gradient variance.')

flags.DEFINE_boolean(
    'log_batch_size_grads', True, 'Whether to log gradient norm for different batch sizes.'
)
flags.DEFINE_integer('batch_size_grads_compute_interval', 500, 'Batch size grads compute interval.')
flags.DEFINE_integer(
    'smallest_batch_size_grad', 64, 'Every power of 2 between smallest and largest.'
)
flags.DEFINE_integer(
    'largest_batch_size_grad', 1024, 'Every power of 2 between smallest and largest.'
)

flags.DEFINE_string('tpu_project', '', 'TPU project.')
flags.DEFINE_string('tpu_name', '', 'TPU name.')

config_flags.DEFINE_config_file(
    'agent', 'jaxrl/agents/bro_minimal/bro_minimal_learner.py', lock_config=False
)

"""
class flags:
    save_dir = './tmp/'
    seed=0
    eval_episodes=1
    eval_interval=2000
    batch_size=int(256)
    max_steps=int(500000)
    replay_buffer_size=int(500000)
    start_training=int(1000)
    updates_per_step=int(2)
    num_seeds=2
    benchmark='dmc'
    env_name='cheetah-run'
    offline_evaluation = False
    width_critic=512
        
FLAGS = flags()
"""


def validate_args():
    def is_power_of_two(x: int):
        return (x & (x - 1)) == 0 and x > 0

    def divisor_or_multiple(x: int, y: int):
        return x % y == 0 or y % x == 0

    if FLAGS.old_target_update_intervals:
        FLAGS.old_target_update_intervals = [int(x) for x in FLAGS.old_target_update_intervals]

    if FLAGS.log_batch_size_grads:
        assert is_power_of_two(FLAGS.smallest_batch_size_grad) and is_power_of_two(
            FLAGS.largest_batch_size_grad
        )

    if FLAGS.log_grad_var and FLAGS.log_batch_size_grads:
        assert divisor_or_multiple(
            FLAGS.grad_var_compute_interval, FLAGS.batch_size_grads_compute_interval
        )
        batch_sizes = np.arange(
            np.log2(FLAGS.smallest_batch_size_grad),
            np.log2(FLAGS.largest_batch_size_grad) + 1,
        )
        batch_sizes = np.power(2, batch_sizes).astype(int)
        assert FLAGS.grad_var_batch_size in batch_sizes


def main(_):
    validate_args()
    config = FLAGS.agent

    MAX_EPISODE_LENGTHS = {
        'shadowhand': 100,
        'mw': 200,
        'dmc': 1000,
        'humanoid_bench': 1000,
    }
    stop_at_offline_return = {
        'acrobot-swingup': 500,
        'cheetah-run': 800,
        'dog-stand': 950,
        'dog-trot': 800,
        'dog-walk': 850,
        'finger-turn': 950,
        'fish-swim': 800,
        'hopper-hop': 400,
        'h1-crawl-v0': 800,
        'h1-pole-v0': 700,
        'h1-stand-v0': 700,
        'h1-walk-v0': 650,
        'humanoid-stand': 900,
        'humanoid-walk': 850,
        'pendulum-swingup': 850,
        'quadruped-run': 800,
        'walker-run': 750,
    }

    env_kw = dict(
        benchmark=FLAGS.benchmark,
        env_name=FLAGS.env_name,
        num_envs=FLAGS.num_seeds,
        max_episode_length=MAX_EPISODE_LENGTHS[FLAGS.benchmark],
    )
    env = make_env(**env_kw, seed=FLAGS.seed)
    if FLAGS.offline_evaluation:
        eval_env = make_env(**env_kw, seed=FLAGS.seed + 42)
    if FLAGS.validation:
        val_env = make_env(**env_kw, seed=FLAGS.seed + 2 * 42)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    mute_warning()

    agent_cls = agents[config['agent_name']]
    agent = agent_cls(
        FLAGS.seed,
        env.observation_space.sample()[0, np.newaxis],
        env.action_space.sample()[0, np.newaxis],
        num_seeds=FLAGS.num_seeds,
        config=config,
    )
    config['model_size'] = agent.get_num_params()
    replay_buffer = ParallelReplayBuffer(
        env.observation_space,
        env.action_space.shape[-1],
        FLAGS.replay_buffer_size,
        num_seeds=FLAGS.num_seeds,
    )
    observations = env.reset()
    if FLAGS.validation:
        val_observations = val_env.reset()
        val_replay_buffer = ParallelReplayBuffer(
            val_env.observation_space,
            val_env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_seeds=FLAGS.num_seeds,
        )

    run_id = init_wandb(FLAGS, config)

    returns_online = np.zeros(FLAGS.num_seeds)
    goals_online = np.zeros(FLAGS.num_seeds)
    counts = np.zeros(FLAGS.num_seeds)
    returns_online_episode = np.zeros(FLAGS.num_seeds)
    goals_online_episode = np.zeros(FLAGS.num_seeds)

    rolling_meter = RollingMeter(FLAGS.log_interval)

    if FLAGS.log_old_target_tderr:
        save_dir = os.path.join(FLAGS.save_dir, run_id)
        os.makedirs(save_dir, exist_ok=True)
        old_target_critic = {
            interval: deepcopy(agent.target_critic)
            for interval in FLAGS.old_target_update_intervals
        }
        n_gradient_steps = 0
        hard_target_recent_update = 0

    start_time = time.time()
    stop = False

    for i in tqdm(range(1, FLAGS.max_steps + 1), disable=not FLAGS.tqdm or is_using_slurm()):
        with Timer('sample_action'):
            if i < FLAGS.start_training:
                actions = env.action_space.sample()
            else:
                actions = agent.sample_actions_o(observations, temperature=1.0)

        with Timer('env_step'):
            next_observations, rewards, terms, truns, goals = env.step(actions)

        with Timer('step_processing'):
            returns_online_episode += rewards
            goals_online_episode += goals

            masks = env.generate_masks(terms, truns)
            # logging online stuff
            if truns.any() or terms.any():
                done = np.logical_or(truns, terms)
                counts = np.where(done, counts + 1, counts)
                goals_online_episode[goals_online_episode > 0.0] = 1.0
                goals_online = np.where(done, goals_online + goals_online_episode, goals_online)
                returns_online = np.where(
                    done, returns_online + returns_online_episode, returns_online
                )
                returns_online_episode = np.where(done, 0, returns_online_episode)
                goals_online_episode = np.where(done, 0, goals_online_episode)

            replay_buffer.insert(observations, actions, rewards, masks, truns, next_observations)
            observations = next_observations
            observations, terms, truns = env.reset_where_done(observations, terms, truns)

        with Timer('validation'):
            if FLAGS.validation and (
                i < FLAGS.start_training or i % FLAGS.validation_interval == 0
            ):
                val_actions = (
                    val_env.action_space.sample()
                    if i < FLAGS.start_training
                    else agent.sample_actions_o(val_observations, temperature=1.0)
                )
                val_next_observations, val_rewards, val_terms, val_truns, val_goals = val_env.step(
                    val_actions
                )
                val_masks = val_env.generate_masks(val_terms, val_truns)
                val_replay_buffer.insert(
                    val_observations,
                    val_actions,
                    val_rewards,
                    val_masks,
                    val_truns,
                    val_next_observations,
                )
                val_observations = val_next_observations
                val_observations, val_terms, val_truns = val_env.reset_where_done(
                    val_observations, val_terms, val_truns
                )

        if i >= FLAGS.start_training:
            all_infos = {}
            update_batches, update_batches_idx = replay_buffer.sample_parallel_multibatch(
                config['batch_size'], config['updates_per_step'], include_idx=True
            )

            with Timer('replay_tderr'):
                if FLAGS.log_replay_tderr and i % FLAGS.replay_compute_interval == 0:
                    replay_batches = replay_buffer.sample_parallel_multibatch(
                        FLAGS.replay_oldnew_batch_size, 1
                    )
                    replay_critic_loss = agent.compute_critic_loss(replay_batches)
                    rolling_meter.add(i, 'replay_critic_loss', replay_critic_loss)

                    q_info = agent.compute_critic_batch(replay_batches)
                    rolling_meter.add_dict(i, prefix_metrics(q_info, 'replay_', sep=''))

                    if config['use_separate_critic']:
                        separate_q_info = agent.compute_separate_critic_batch(replay_batches)
                        rolling_meter.add_dict(
                            i, prefix_metrics(separate_q_info, 'sep_replay_', sep='')
                        )

                    old_data_batches = replay_buffer.sample_earliest_multibatch(
                        FLAGS.replay_oldnew_batch_size, 1
                    )
                    old_data_critic_losses = agent.compute_critic_loss(old_data_batches)
                    rolling_meter.add(i, 'old_data_critic_loss', old_data_critic_losses)

                    new_data_batches = replay_buffer.sample_latest_multibatch(
                        FLAGS.replay_oldnew_batch_size, 1
                    )
                    new_data_critic_losses = agent.compute_critic_loss(new_data_batches)
                    rolling_meter.add(i, 'new_data_critic_loss', new_data_critic_losses)

                    if config['use_separate_critic']:
                        replay_separate_critic_loss = agent.compute_critic_loss(
                            replay_batches, critic=agent.separate_critic
                        )
                        rolling_meter.add(
                            i, 'replay_separate_critic_loss', replay_separate_critic_loss
                        )

                        old_data_separate_critic_losses = agent.compute_critic_loss(
                            old_data_batches, critic=agent.separate_critic
                        )
                        rolling_meter.add(
                            i, 'old_data_separate_critic_loss', old_data_separate_critic_losses
                        )

                        new_data_separate_critic_losses = agent.compute_critic_loss(
                            new_data_batches, critic=agent.separate_critic
                        )
                        rolling_meter.add(
                            i, 'new_data_separate_critic_loss', new_data_separate_critic_losses
                        )

            with Timer('grad_var'):
                if FLAGS.log_grad_var and i % FLAGS.grad_var_compute_interval == 0:
                    # subsample from training batch
                    grad_var_idx = np.random.choice(
                        update_batches_idx.flatten(),
                        size=(config['updates_per_step'], FLAGS.grad_var_batch_size),
                    )
                    grad_var_batches = replay_buffer.get_at_index(grad_var_idx)
                    grad_var_infos = agent.compute_critic_grad_var(grad_var_batches)
                    grad_var_norm = grad_var_infos['sum_critic_grad_var']
                    adam_grad_var_infos = agent.get_adam_critic_grad_var()
                    rolling_meter.add_dict(i, grad_var_infos | adam_grad_var_infos)

            with Timer('batch_size_grads'):
                if FLAGS.log_batch_size_grads and i % FLAGS.batch_size_grads_compute_interval == 0:
                    batch_size_grad_norms = {}
                    batch_sizes = np.arange(
                        np.log2(FLAGS.smallest_batch_size_grad),
                        np.log2(FLAGS.largest_batch_size_grad) + 1,
                    )
                    batch_sizes = np.power(2, batch_sizes).astype(int)
                    for batch_size in batch_sizes:
                        batch = replay_buffer.sample_parallel_multibatch(batch_size, 1)
                        batch_size_grad_norms[f'grad_norm_b{batch_size}'] = (
                            agent.compute_critic_grad_norm(batch)
                        )
                        if config['use_separate_critic']:
                            batch_size_grad_norms[f'sep_grad_norm_b{batch_size}'] = (
                                agent.compute_separate_critic_grad_norm(batch)
                            )

                    rolling_meter.add_dict(i, batch_size_grad_norms)

                    if FLAGS.log_grad_var and i % FLAGS.grad_var_compute_interval == 0:
                        rolling_meter.add(
                            i,
                            'relative_grad_var',
                            grad_var_norm
                            / batch_size_grad_norms[f'grad_norm_b{FLAGS.grad_var_batch_size}'] ** 2,
                        )

            with Timer('update'):
                should_hard_target_update = (
                    config.hard_target_update
                    and n_gradient_steps - hard_target_recent_update
                    >= config.hard_target_update_interval
                )
                update_infos = agent.update(
                    update_batches,
                    config['updates_per_step'],
                    i,
                    should_hard_target_update,
                )
                n_gradient_steps += config['updates_per_step']
                if should_hard_target_update:
                    hard_target_recent_update = n_gradient_steps
                all_infos.update(update_infos)

            with Timer('replay_tderr'):
                if FLAGS.log_replay_tderr and i % FLAGS.replay_compute_interval == 0:
                    postupd_replay_critic_loss = agent.compute_critic_loss(replay_batches)
                    rolling_meter.add(i, 'postupd_replay_critic_loss', postupd_replay_critic_loss)

                    postupd_q_info = agent.compute_critic_batch(replay_batches)
                    rolling_meter.add_dict(
                        i, prefix_metrics(postupd_q_info, 'postupd_replay_', sep='')
                    )

                    if config['use_separate_critic']:
                        separate_postupd_q_info = agent.compute_separate_critic_batch(
                            replay_batches
                        )
                        rolling_meter.add_dict(
                            i,
                            prefix_metrics(separate_postupd_q_info, 'sep_postupd_replay_', sep=''),
                        )

            with Timer('validation'):
                if FLAGS.validation and i % FLAGS.validation_interval == 0:
                    val_batches = val_replay_buffer.sample_parallel_multibatch(
                        config['batch_size'], 1
                    )
                    val_critic_loss = agent.compute_critic_loss(val_batches)
                    rolling_meter.add(i, 'validation_critic_loss', val_critic_loss)

                    if config['use_separate_critic']:
                        sep_val_critic_loss = agent.compute_critic_loss(
                            val_batches, critic=agent.separate_critic
                        )
                        rolling_meter.add(i, 'sep_validation_critic_loss', sep_val_critic_loss)

            with Timer('offline_eval'):
                if FLAGS.offline_evaluation and i % FLAGS.eval_interval == 0:
                    eval_stats = eval_env.evaluate(
                        agent, num_episodes=FLAGS.eval_episodes, temperature=0.0
                    )
                    if eval_stats['return'].mean() > stop_at_offline_return.get(
                        FLAGS.env_name, float('inf')
                    ):
                        stop = True
                    all_infos.update(eval_stats)

            if config.use_reset and i % (config.reset_interval // config.updates_per_step) == 0:
                agent.reset()

            # use post-update target
            with Timer('old_target_tderr'):
                if FLAGS.log_old_target_tderr:
                    for interval in FLAGS.old_target_update_intervals:
                        if (i + interval) % FLAGS.old_target_compute_interval == 0:
                            old_target_critic[interval] = old_target_critic[interval].replace_(
                                params=agent.target_critic.params
                            )
                    if i % FLAGS.old_target_compute_interval == 0:
                        for interval in FLAGS.old_target_update_intervals:
                            old_target_critic_loss = agent.compute_critic_loss(
                                update_batches, target_critic=old_target_critic[interval]
                            )
                            rolling_meter.add(
                                i, f'old_target_critic_loss_{interval}', old_target_critic_loss
                            )

            if i % FLAGS.log_interval == 0:
                # Log online stats
                counts = np.where(counts == 0.0, 1e-8, counts)
                infos_online_eval = {
                    'goal_online': goals_online / counts,
                    'return_online': returns_online / counts,
                }

                all_infos.update(infos_online_eval)
                returns_online = np.zeros(FLAGS.num_seeds)
                goals_online = np.zeros(FLAGS.num_seeds)
                counts = np.zeros(FLAGS.num_seeds)

                all_infos.update(rolling_meter.get_log_data())
                log_seeds_to_wandb(i, all_infos)

                end_time = time.time()
                times_dict = Timer.get_totals()
                times_dict['total'] = end_time - start_time
                start_time = end_time
                Timer.reset()

                general_info = {
                    'grad_steps': n_gradient_steps,
                    **prefix_metrics(times_dict, 'time'),
                }
                if gpu_id is not None:
                    general_info['System/GPU ID'] = int(gpu_id)
                if tpu_id is not None:
                    general_info['System/TPU ID'] = int(tpu_id)
                wandb.log(general_info, step=i)

            if stop:
                return


if __name__ == '__main__':
    app.run(main)
