import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import jax
import functools


def preprocess_state(state_dict):
    return np.concatenate(
        (
            state_dict['observation'],
            state_dict['achieved_goal'],
            state_dict['desired_goal'],
        )
    )


class make_env:
    def __init__(self, benchmark, env_name, num_envs=5, seed=0, max_episode_length=1000):
        assert benchmark in ['mw', 'shadowhand', 'dmc', 'humanoid_bench']
        np.random.seed(seed)
        random.seed(seed)
        self.num_envs = num_envs
        seeds = np.random.randint(0, 1e6, (self.num_envs))
        self.seeds = seeds
        self.benchmark = benchmark
        if benchmark == 'shadowhand':
            import gymnasium_robotics
            from gymnasium.wrappers import FlattenObservation, RescaleAction

            gym.register_envs(gymnasium_robotics)

            def make_single_env(task, seed):
                env = gym.make(task, max_episode_steps=max_episode_length)
                # env = FlattenObservation(env)
                env = RescaleAction(env, min_action=-1.0, max_action=1.0)
                env.action_space.seed(int(seed))
                return env

            self.envs = [make_single_env(env_name, seed) for seed in seeds]
            self.goal_name = 'is_success'
        if benchmark == 'mw':
            from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

            constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
            self.envs = [constructor(seed=int(seed)) for seed in seeds]
            self.goal_name = 'success'
        if benchmark == 'dmc':
            from jaxrl.envs.dmc_gym import _make_env_dmc

            env_fns = [lambda i=i: _make_env_dmc(env_name, 1) for i in seeds]
            self.envs = [env_fn() for env_fn in env_fns]
            self.goal_name = 'success'
        if benchmark == 'humanoid_bench':
            import humanoid_bench
            from humanoid_bench.env import ROBOTS, TASKS

            assert env_name in [
                'h1-walk-v0',
                'h1-stand-v0',
                'h1-run-v0',
                'h1-stair-v0',
                'h1-crawl-v0',
                'h1-pole-v0',
                'h1-maze-v0',
                'h1-slide-v0',
                'h1-hurdle-v0',
                'h1-reach-v0',
                'h1-balance_simple-v0',
                'h1-balance_hard-v0',
                'h1-sit_simple-v0',
                'h1-sit_hard-v0',
                'h1hand-walk-v0',
                'h1hand-stand-v0',
                'h1hand-run-v0',
                'h1hand-stair-v0',
                'h1hand-crawl-v0',
                'h1hand-pole-v0',
                'h1hand-slide-v0',
                'h1hand-hurdle-v0',
                'h1hand-maze-v0',
                'h1hand-sit_simple-v0',
                'h1hand-sit_hard-v0',
                'h1hand-balance_simple-v0',
                'h1hand-balance_hard-v0',
                'h1hand-reach-v0',
                'h1hand-spoon-v0',
                'h1hand-window-v0',
                'h1hand-insert_small-v0',
                'h1hand-insert_normal-v0',
                'h1hand-kitchen-v0',
                'h1hand-push-v0',
                'h1hand-cabinet-v0',
                'h1hand-bookshelf_simple-v0',
                'h1hand-bookshelf_hard-v0',
                'h1hand-cube-v0',
                'h1hand-truck-v0',
                'h1strong-highbar_hard-v0',
                'h1hand-package-v0',
                'h1hand-powerlift-v0',
                'h1hand-room-v0',
                'h1hand-door-v0',
                'h1hand-basketball-v0',
                'h1hand-kitchen-v0',
            ]
            self.envs = [gym.make(env_name, autoreset=False) for i in seeds]
            self.goal_name = 'success'

        self.timesteps = np.zeros(len(self.envs))
        self.max_episode_length = max_episode_length

        self.action_space = spaces.Box(
            low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
            high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
            shape=(len(self.envs), self.envs[0].action_space.shape[0]),
            dtype=self.envs[0].action_space.dtype,
        )

        if benchmark != 'shadowhand':
            self.observation_space = spaces.Box(
                low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                dtype=self.envs[0].observation_space.dtype,
            )
        else:
            low = np.concatenate(
                (
                    self.envs[0].observation_space['observation'].low,
                    self.envs[0].observation_space['achieved_goal'].low,
                    self.envs[0].observation_space['desired_goal'].low,
                )
            )
            high = np.concatenate(
                (
                    self.envs[0].observation_space['observation'].high,
                    self.envs[0].observation_space['achieved_goal'].high,
                    self.envs[0].observation_space['desired_goal'].high,
                )
            )

            self.observation_space = spaces.Box(
                low=low[None].repeat(len(self.envs), axis=0),
                high=high[None].repeat(len(self.envs), axis=0),
                shape=(len(self.envs), high.shape[0]),
                dtype=self.envs[0].observation_space['observation'].dtype,
            )

    def _reset_idx(self, idx):
        seed = np.random.randint(0, 1e7)
        self.timesteps[idx] = 0
        state, _ = self.envs[idx].reset(seed=seed)
        if self.benchmark == 'shadowhand':
            state = preprocess_state(state)
        return state

    def reset_where_done(self, states, terminals, truncates):
        for j, (terminal, truncate) in enumerate(zip(terminals, truncates)):
            if (terminal == True) or (truncate == True):
                states[j], terminals[j], truncates[j] = self._reset_idx(j), False, False
        return states, terminals, truncates

    def reset(self):
        states = []
        for i, env in enumerate(self.envs):
            states.append(self._reset_idx(i))
        return np.stack(states)

    def generate_masks(self, terminals, truncates):
        masks = []
        for terminal, truncate in zip(terminals, truncates):
            if not terminal or truncate:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        return np.array(masks)

    def step(self, actions):
        self.timesteps += 1
        states, rewards, terminals, truncates, goals = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminal, _, info = env.step(action)
            if self.benchmark == 'shadowhand':
                state = preprocess_state(state)
            if self.timesteps[i] == self.max_episode_length:
                truncate = True
            else:
                truncate = False
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            truncates.append(truncate)
            if self.benchmark == 'humanoid_bench':
                goals.append(0)
            else:
                goals.append(info[self.goal_name])
        states = np.stack(states)
        return (
            np.stack(states),
            np.stack(rewards),
            np.stack(terminals),
            np.stack(truncates),
            np.stack(goals),
        )

    def _compute_mean_q(self, agent, obs, actions):
        """(num_envs, T, ...) -> (num_envs, T)"""

        @functools.partial(jax.vmap, in_axes=(1, 1))
        def compute(obs_t, act_t):
            q1, q2 = agent.compute_critic(obs_t, act_t)
            return 0.5 * (q1.mean(-1) + q2.mean(-1))

        return compute(obs, actions).T

    def _calculate_returns_to_go(self, rewards_traj, log_probs_traj, discount, temp, bootstrap):
        values = np.zeros_like(rewards_traj)
        for i in reversed(range(rewards_traj.shape[-1])):
            values[..., i] = (
                rewards_traj[..., i] - temp * log_probs_traj[..., i] + discount * bootstrap
            )
            bootstrap = values[..., i]
        return values

    def evaluate_nonterminating(self, agent, num_episodes=5, temperature=0.0):
        temp = agent.compute_temp()
        max_T = self.max_episode_length
        d_obs = self.observation_space.shape[-1]
        d_act = self.action_space.shape[-1]

        traj = {
            'obs': np.full((num_episodes, self.num_envs, max_T, d_obs), np.nan),
            'actions': np.full((num_episodes, self.num_envs, max_T, d_act), np.nan),
            'log_probs': np.full((num_episodes, self.num_envs, max_T), np.nan),
            'rewards': np.full((num_episodes, self.num_envs, max_T), np.nan),
            'next_obs': np.full((num_episodes, self.num_envs, max_T, d_obs), np.nan),
            'returns_to_go': np.full((num_episodes, self.num_envs, max_T), np.nan),
        }

        goals = []
        last_step_success = []
        returns_eval = []
        losses = []

        for n in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(self.num_envs)
            goal = 0.0

            for t in range(max_T):
                actions = agent.sample_actions(observations, temperature=temperature)
                log_probs = agent.log_prob(observations, actions, temperature=1.0)
                next_obs, rewards, terms, truns, success = self.step(actions)

                traj['obs'][n, :, t] = observations
                traj['actions'][n, :, t] = actions
                traj['log_probs'][n, :, t] = log_probs
                traj['rewards'][n, :, t] = rewards
                traj['next_obs'][n, :, t] = next_obs

                returns += rewards
                goal += success / max_T
                observations = next_obs

            goal[goal > 0] = 1.0
            goals.append(goal)
            last_step_success.append(success)
            returns_eval.append(returns)

            rewards = traj['rewards'][n]
            log_probs = traj['log_probs'][n]
            returns_to_go = self._calculate_returns_to_go(
                rewards, log_probs, agent.discount, temp, np.ones(self.num_envs)
            )
            traj['returns_to_go'][n] = returns_to_go

            obs = traj['obs'][n]
            actions = traj['actions'][n]
            q_vals = self._compute_mean_q(agent, obs, actions)
            losses.append(np.abs(q_vals - returns_to_go))

        qs = np.stack(
            [
                self._compute_mean_q(agent, traj['obs'][n], traj['actions'][n])
                for n in range(num_episodes)
            ]
        )
        returns_to_go = traj['returns_to_go']
        losses = np.stack(losses)

        frac_to_log = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        idx_to_log = np.minimum(frac_to_log * (max_T - 1), max_T - 1).astype(int)

        res_dict = {
            'goal': np.array(goals).mean(axis=0),
            'last_step_success': np.array(last_step_success).mean(axis=0),
            'return': np.array(returns_eval).mean(axis=0),
            'length': np.full(self.num_envs, max_T),
            'eval_q_avg': np.nanmean(qs, axis=(0, -1)),
            'gt_return_to_go_avg': np.nanmean(returns_to_go, axis=(0, -1)),
            'gt_loss_avg': np.nanmean(losses, axis=(0, -1)),
        }
        for frac, idx in zip(frac_to_log, idx_to_log):
            res_dict[f'eval_q_{float(frac)}'] = qs[:, :, idx].mean(axis=0)
            res_dict[f'gt_return_to_go_{float(frac)}'] = returns_to_go[:, :, idx].mean(axis=0)
            res_dict[f'gt_loss_{float(frac)}'] = losses[:, :, idx].mean(axis=0)

        return res_dict

    def evaluate_terminating(self, agent, num_episodes=5, temperature=0.0):
        temp = agent.compute_temp()
        max_T = self.max_episode_length
        d_obs = self.observation_space.shape[-1]
        d_act = self.action_space.shape[-1]

        traj = {
            'obs': np.full((num_episodes, self.num_envs, max_T, d_obs), np.nan),
            'actions': np.full((num_episodes, self.num_envs, max_T, d_act), np.nan),
            'log_probs': np.full((num_episodes, self.num_envs, max_T), np.nan),
            'rewards': np.full((num_episodes, self.num_envs, max_T), np.nan),
            'next_obs': np.full((num_episodes, self.num_envs, max_T, d_obs), np.nan),
            'returns_to_go': np.full((num_episodes, self.num_envs, max_T), np.nan),
        }
        lengths = np.zeros((num_episodes, self.num_envs), dtype=int)
        returns = np.zeros((num_episodes, self.num_envs))
        truncateds = np.zeros((num_episodes, self.num_envs))
        n_trajs = np.zeros(self.num_envs, dtype=int)
        timesteps = np.zeros(self.num_envs, dtype=int)

        observations = self.reset()

        while n_trajs.min() < num_episodes:
            actions = agent.sample_actions(observations, temperature=temperature)
            log_probs = agent.log_prob(observations, actions, temperature=1.0)
            next_obs, rewards, terms, truns, success = self.step(actions)

            for e in range(self.num_envs):
                n_traj = n_trajs[e]
                timestep = timesteps[e]
                if n_traj >= num_episodes:
                    continue

                traj['obs'][n_traj, e, timestep] = observations[e]
                traj['actions'][n_traj, e, timestep] = actions[e]
                traj['log_probs'][n_traj, e, timestep] = log_probs[e]
                traj['rewards'][n_traj, e, timestep] = rewards[e]
                traj['next_obs'][n_traj, e, timestep] = next_obs[e]
                timesteps[e] += 1

                if terms[e] or truns[e]:
                    lengths[n_traj, e] = timesteps[e]
                    returns[n_traj, e] = np.sum(traj['rewards'][n_traj, e, : timesteps[e]])
                    truncateds[n_traj, e] = truns[e]
                    n_trajs[e] += 1
                    timesteps[e] = 0

            observations, terms, truns = self.reset_where_done(next_obs, terms, truns)
            observations = np.array(observations)

        # Compute bootstrap targets
        for n in range(num_episodes):
            final_obs = np.stack(
                [traj['next_obs'][n, e, lengths[n, e] - 1] for e in range(self.num_envs)]
            )
            final_actions = agent.sample_actions(final_obs, temperature=temperature)
            final_log_probs = agent.log_prob(final_obs, final_actions, temperature=1.0)
            q1, q2 = agent.compute_target_critic(final_obs, final_actions)
            bootstrap = ((q1 + q2) / 2).mean(axis=-1) - temp * final_log_probs
            bootstrap *= truncateds[n]

            for e in range(self.num_envs):
                T = lengths[n, e]
                rewards = traj['rewards'][n, e, :T]
                log_probs = traj['log_probs'][n, e, :T]
                traj['returns_to_go'][n, e, :T] = self._calculate_returns_to_go(
                    rewards, log_probs, agent.discount, temp[e], bootstrap[e]
                )

        # Compute q-values and losses
        qs = []
        returns_to_go = []
        losses = []

        for n in range(num_episodes):
            obs = traj['obs'][n]
            actions = traj['actions'][n]
            returns_to_go_ = traj['returns_to_go'][n]
            qs_ = np.array(self._compute_mean_q(agent, obs, actions))
            for e in range(self.num_envs):
                qs_[e, lengths[n, e] :] = np.nan
            qs.append(qs_)
            losses.append(np.abs(qs_ - returns_to_go_))
            returns_to_go.append(returns_to_go_)

        qs = np.stack(qs)
        returns_to_go = np.stack(returns_to_go)
        losses = np.stack(losses)

        # Logging specific timesteps
        frac_to_log = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        frac_to_log = frac_to_log[frac_to_log * (max_T - 1) < max_T]
        idx_to_log = (frac_to_log * (max_T - 1)).astype(int)

        res_dict = {
            'goal': np.zeros(self.num_envs),
            'last_step_success': np.zeros(self.num_envs),
            'return': returns.mean(axis=0),
            'length': lengths.mean(axis=0),
            'eval_q_avg': np.nanmean(qs, axis=(0, -1)),
            'gt_return_to_go_avg': np.nanmean(returns_to_go, axis=(0, -1)),
            'gt_loss_avg': np.nanmean(losses, axis=(0, -1)),
        }
        for frac, idx in zip(frac_to_log, idx_to_log):
            res_dict[f'eval_q_{float(frac)}'] = qs[:, :, idx].mean(axis=0)
            res_dict[f'gt_return_to_go_{float(frac)}'] = returns_to_go[:, :, idx].mean(axis=0)
            res_dict[f'gt_loss_{float(frac)}'] = losses[:, :, idx].mean(axis=0)

        return res_dict

    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        if self.benchmark in ['humanoid_bench', 'shadowhand']:
            return self.evaluate_terminating(agent, num_episodes, temperature)
        else:
            return self.evaluate_nonterminating(agent, num_episodes, temperature)
