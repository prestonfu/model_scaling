import os
import glob

os.environ['MUJOCO_GL'] = 'egl'

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

import numpy as np
import wandb
from absl import app, flags
from jaxrl.agents import agents
from jaxrl.envs.new_env import make_env
from jaxrl.replay_buffer import ParallelReplayBuffer, NStepParallelReplayBuffer
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
flags.DEFINE_boolean('save_checkpoints', False, 'Save checkpoints.')
flags.DEFINE_integer('save_interval', 100_000, 'Checkpoint saving interval.')
flags.DEFINE_integer('max_steps', int(2000000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(2500), 'Number of training steps to start training.')
flags.DEFINE_integer('num_seeds', 5, 'Number of parallel seeds to run.')
flags.DEFINE_string('benchmark', 'dmc', 'Environment name.')
flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('offline_evaluation', True, 'Do offline eval.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm.')
flags.DEFINE_string('wandb_entity', 'prestonfu', 'Wandb entity.')
flags.DEFINE_string('wandb_project', 'BRO_model_scaling', 'Wandb project.')
flags.DEFINE_string('wandb_tag', 'debug', 'Wandb tag.')
flags.DEFINE_boolean('stop_at_success', True, 'Stop at threshold given in stop_at_offline_return.')
flags.DEFINE_integer('n_step', 1, 'N-step TD.')
flags.DEFINE_boolean('use_nstep_replay_buffer', False, '(buggy) Use n-step replay buffer.')

# saving and loading
flags.DEFINE_string('load_checkpoint_dir', None, 'Path within save_dir / wandb_tag directory.')
flags.DEFINE_integer('load_checkpoint_step', 100_000, 'Step of the checkpoint to load.')
flags.DEFINE_bool('load_actor_only', False, 'Load actor only.')
flags.DEFINE_bool('load_critic_only', False, 'Load critic only.')
flags.DEFINE_bool('load_separate_critic_only', False, 'Load separate critic only.')

flags.DEFINE_boolean('validation', True, 'Whether to log validation.')
flags.DEFINE_integer('validation_interval', 10, 'Validation frequency (env steps).')

flags.DEFINE_string('tpu_project', '', 'TPU project.')
flags.DEFINE_string('tpu_name', '', 'TPU name.')

config_flags.DEFINE_config_file(
    'agent', 'jaxrl/agents/bro_minimal/bro_minimal_learner.py', lock_config=False
)


def validate_args():
    if FLAGS.n_step > 1:
        assert FLAGS.use_nstep_replay_buffer
        assert not FLAGS.validation or FLAGS.validation_interval == 1, 'Not implemented yet'


def find_checkpoint_path(checkpoint_dir, width_critic=None):
    # assume that checkpoint_dir is the save_dir / wandb_tag directory
    # look for the run_id that matches the current seed
    if width_critic is None:
        width_critic = FLAGS.agent.width_critic
    # run_spec = f"utd{FLAGS.agent.updates_per_step}_bs{FLAGS.agent.batch_size}_critic{width_critic}"
    matching_run = glob.glob(
        os.path.join(
            checkpoint_dir,  # pass in full checkpoint dir including wandb id
            f'step_{FLAGS.load_checkpoint_step}',
        )
    )
    assert len(matching_run) >= 1, breakpoint()  # f"No matching run found"
    checkpoint_path = matching_run[0]
    assert os.path.exists(checkpoint_path)
    return checkpoint_path


def main(_):
    validate_args()
    config = FLAGS.agent
    if config['target_noise_kind'] == 'pretrained':
        config['target_pretrained_path'] = find_checkpoint_path(
            config['target_pretrained_path'], width_critic=config['target_pretrained_critic_width']
        )

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
    # load checkpoint if specified
    if FLAGS.load_checkpoint_dir is not None:
        assert os.path.exists(FLAGS.load_checkpoint_dir)
        checkpoint_path = find_checkpoint_path(FLAGS.load_checkpoint_dir)
        if FLAGS.load_actor_only:
            agent.load_actor(checkpoint_path)
        elif FLAGS.load_critic_only:
            agent.load_critic(checkpoint_path)
        elif FLAGS.load_separate_critic_only:
            agent.load_separate_critic(checkpoint_path)
        else:
            agent.load(checkpoint_path)
    config['model_size'] = agent.get_num_params()

    if not FLAGS.use_nstep_replay_buffer:
        replay_buffer = ParallelReplayBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_seeds=FLAGS.num_seeds,
            gamma=FLAGS.agent.discount,
        )
    else:
        replay_buffer = NStepParallelReplayBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_seeds=FLAGS.num_seeds,
            n_step=FLAGS.n_step,
            gamma=FLAGS.agent.discount,
        )

    observations = env.reset()
    if FLAGS.validation:
        val_observations = val_env.reset()
        if not FLAGS.use_nstep_replay_buffer:
            val_replay_buffer = ParallelReplayBuffer(
                val_env.observation_space,
                val_env.action_space.shape[-1],
                FLAGS.replay_buffer_size,
                num_seeds=FLAGS.num_seeds,
                gamma=FLAGS.agent.discount,
            )
        else:
            val_replay_buffer = NStepParallelReplayBuffer(
                val_env.observation_space,
                val_env.action_space.shape[-1],
                FLAGS.replay_buffer_size,
                num_seeds=FLAGS.num_seeds,
                n_step=FLAGS.n_step,
                gamma=FLAGS.agent.discount,
            )

    returns_online = np.zeros(FLAGS.num_seeds)
    goals_online = np.zeros(FLAGS.num_seeds)
    counts = np.zeros(FLAGS.num_seeds)
    returns_online_episode = np.zeros(FLAGS.num_seeds)
    goals_online_episode = np.zeros(FLAGS.num_seeds)

    rolling_meter = RollingMeter(FLAGS.log_interval)

    run_id = init_wandb(FLAGS, config)

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_tag, f'{run_id}')
    os.makedirs(save_dir, exist_ok=True)
    os.chmod(save_dir, 0o777)  # Set directory permissions to rwxrwxrwx

    n_gradient_steps = 0
    hard_target_recent_update = 0

    start_time = time.time()
    stop = False
    stop_threshold = stop_at_offline_return.get(FLAGS.env_name, float('inf'))

    for i in tqdm(range(1, FLAGS.max_steps + 1), disable=not FLAGS.tqdm or is_using_slurm()):
        # save the agent
        if FLAGS.save_checkpoints and i % FLAGS.save_interval == 0:
            agent.save(os.path.join(save_dir, f'step_{i}'))

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

            with Timer('validation'):
                if FLAGS.validation and i % FLAGS.validation_interval == 0:
                    val_batches = val_replay_buffer.sample_parallel_multibatch(
                        config['batch_size'], 1
                    )
                    val_q_info = agent.compute_critic_batch(val_batches)
                    rolling_meter.add_dict(i, prefix_metrics(val_q_info, 'val_', sep=''))
                    val_critic_loss = agent.compute_critic_loss(val_batches)
                    rolling_meter.add(i, 'validation_critic_loss', val_critic_loss)

                    if config['use_separate_critic']:
                        sep_val_critic_loss = agent.compute_critic_loss(
                            val_batches, critic=agent.separate_critic
                        )
                        rolling_meter.add(i, 'sep_validation_critic_loss', sep_val_critic_loss)

            quotient_counts = np.where(counts == 0.0, 1e-8, counts)
            infos_online_eval = {
                'goal_online': goals_online / quotient_counts,
                'return_online': returns_online / quotient_counts,
            }

            with Timer('offline_eval'):
                if FLAGS.offline_evaluation and i % FLAGS.eval_interval == 0:
                    eval_stats = eval_env.evaluate(
                        agent, num_episodes=FLAGS.eval_episodes, temperature=0.0
                    )
                    if (
                        FLAGS.stop_at_success
                        and infos_online_eval['return_online'].mean() > stop_threshold
                        and eval_stats['return'].mean() > stop_threshold
                    ):
                        stop = True
                    all_infos.update(eval_stats)

            if config.use_reset and i % (config.reset_interval // config.updates_per_step) == 0:
                agent.reset()

            if i % FLAGS.log_interval == 0:
                # Log online stats
                all_infos.update(infos_online_eval)
                returns_online = np.zeros(FLAGS.num_seeds)
                goals_online = np.zeros(FLAGS.num_seeds)
                counts = np.zeros(FLAGS.num_seeds)

                all_infos.update(rolling_meter.get_log_data())
                try:
                    log_seeds_to_wandb(i, all_infos)
                except Exception as e:
                    print(f'Error logging to wandb: {e}')
                    for k, v in all_infos.items():
                        print(f'{k}: {v}')
                    raise e

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
