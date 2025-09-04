import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import ml_collections
from typing import Tuple

from jaxrl.agents.bro_minimal import temperature
from jaxrl.agents.bro_minimal.actor import update as update_actor

from jaxrl.agents.bro_minimal.critic import (
    update as update_critic,
    update_quantile as update_critic_quantile,
    target_update,
    hard_target_update,
    compute_loss as _compute_critic_loss,
    compute_quantile_loss as _compute_quantile_critic_loss,
    compute_critic_grad_var as _compute_critic_grad_var,
    compute_quantile_critic_grad_var as _compute_quantile_critic_grad_var,
    compute_critic_grad_norm as _compute_critic_grad_norm,
    compute_quantile_critic_grad_norm as _compute_quantile_critic_grad_norm,
)

from jaxrl.replay_buffer import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import (
    InfoDict,
    Model,
    MLPClassic,
    ModelWrapper,
    GaussianNoiseModel,
    NetworkNoiseModel,
    PRNGKey,
)
from jaxrl.utils import prefix_metrics


@functools.partial(
    jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None)
)
def _update(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    separate_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    distributional: bool,
    quantile_taus: jnp.ndarray,
    pessimism: float,
    use_hard_target_update: bool,
    use_separate_critic: bool,
):
    rng, key = jax.random.split(rng)
    critic_input_dict = dict(
        key=key,
        actor=actor,
        temp=temp,
        batch=batch,
        discount=discount,
        pessimism=pessimism,
    )
    if distributional:
        critic_input_dict = dict(
            **critic_input_dict,
            target_quantile_critic=target_critic,
            taus=quantile_taus,
        )
        new_critic, critic_info = update_critic_quantile(
            **critic_input_dict, quantile_critic=critic
        )
        new_separate_critic, separate_critic_info = update_critic_quantile(
            **critic_input_dict,
            quantile_critic=separate_critic,
        )
    else:
        critic_input_dict = dict(
            **critic_input_dict,
            target_critic=target_critic,
        )
        new_critic, critic_info = update_critic(**critic_input_dict, critic=critic)
        new_separate_critic, separate_critic_info = update_critic(
            **critic_input_dict, critic=separate_critic
        )
    if not use_hard_target_update:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic  # hard target update is done outside
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key, actor, new_critic, temp, batch, pessimism, distributional
    )
    rng, key = jax.random.split(rng)
    new_temp, alpha_info = temperature.update_temperature(
        temp, actor_info['entropy'], target_entropy
    )

    info = {**critic_info, **actor_info, **alpha_info}
    if use_separate_critic:
        info.update(prefix_metrics(separate_critic_info, 'separate_', sep=''))

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_separate_critic,
        new_temp,
        info,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        'discount',
        'tau',
        'target_entropy',
        'distributional',
        'pessimism',
        'use_hard_target_update',
        'num_updates',
        'use_separate_critic',
    ),
)
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    separate_critic: Model,
    temp: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    distributional: bool,
    quantile_taus: jnp.ndarray,
    pessimism: bool,
    use_hard_target_update: bool,
    use_separate_critic: bool,
    step: int,
    num_updates: int,
):
    def one_step(i, state):
        step, rng, actor, critic, target_critic, separate_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_separate_critic, new_temp, info = (
            _update(
                rng,
                actor,
                critic,
                target_critic,
                separate_critic,
                temp,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches),
                discount,
                tau,
                target_entropy,
                distributional,
                quantile_taus,
                pessimism,
                use_hard_target_update,
                use_separate_critic,
            )
        )
        return (
            step,
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_separate_critic,
            new_temp,
            info,
        )

    step, rng, actor, critic, target_critic, separate_critic, temp, info = one_step(
        0, (step, rng, actor, critic, target_critic, separate_critic, temp, {})
    )
    return jax.lax.fori_loop(
        1,
        num_updates,
        one_step,
        (step, rng, actor, critic, target_critic, separate_critic, temp, info),
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None))
def _compute_critic_loss_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
):
    rng, key = jax.random.split(rng)
    return _compute_critic_loss(key, actor, critic, target_critic, temp, batch, discount, pessimism)


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))
def _compute_quantile_loss_jit(
    rng: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)
    return _compute_quantile_critic_loss(
        key,
        actor,
        quantile_critic,
        target_quantile_critic,
        temp,
        batch,
        discount,
        pessimism,
        taus,
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None))
def _compute_critic_grad_var_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
):
    rng, key = jax.random.split(rng)
    return _compute_critic_grad_var(
        key, actor, critic, target_critic, temp, batch, discount, pessimism
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))
def _compute_quantile_critic_grad_var_jit(
    rng: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)
    return _compute_quantile_critic_grad_var(
        key,
        actor,
        quantile_critic,
        target_quantile_critic,
        temp,
        batch,
        discount,
        pessimism,
        taus,
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None))
def _compute_critic_grad_norm_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
):
    rng, key = jax.random.split(rng)
    return _compute_critic_grad_norm(
        key, actor, critic, target_critic, temp, batch, discount, pessimism
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))
def _compute_quantile_critic_grad_norm_jit(
    rng: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)
    return _compute_quantile_critic_grad_norm(
        key,
        actor,
        quantile_critic,
        target_quantile_critic,
        temp,
        batch,
        discount,
        pessimism,
        taus,
    )


@jax.jit
@jax.vmap
def _compute_temp_jit(temp: Model):
    return temp.apply({'params': temp.params})


@jax.jit
@jax.vmap
def _compute_critic_jit(critic: Model, observations, actions):
    return critic.apply({'params': critic.params}, observations, actions)


@jax.jit
@jax.vmap
def _compute_target_critic_jit(target_critic: ModelWrapper, observations, actions, rng: PRNGKey):
    rng, key = jax.random.split(rng)
    return target_critic.apply({'params': target_critic.params}, observations, actions, key=key)


@jax.jit
@jax.vmap
def _compute_critic_batch_jit(critic: Model, batch: Batch):
    q1, q2 = critic(batch.observations, batch.actions)
    return {'q1': q1.mean(), 'q2': q2.mean()}


class BROMinimal(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        num_seeds: int,
        config: ml_collections.ConfigDict,
    ) -> None:
        assert config.target_noise_kind in ['none', 'gaussian', 'network']
        if config.target_noise_kind != 'none':
            assert config.target_noise > 0.0
        else:
            assert config.target_noise == 0.0

        self.distributional = config.distributional
        self.n_quantiles = config.n_quantiles
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.pessimism = config.pessimism
        self.use_hard_target_update = config.hard_target_update
        self.use_separate_critic = config.use_separate_critic
        quantile_taus = jnp.arange(0, config.n_quantiles + 1) / config.n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)[None, ...]
        self.seeds = jnp.arange(seed, seed + num_seeds)
        self.target_entropy = (
            -self.action_dim / 2 if config.target_entropy is None else config.target_entropy
        )
        self.tau = config.tau
        self.discount = config.discount
        self.num_seeds = num_seeds
        output_nodes = self.n_quantiles if self.distributional else 1

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
            actor_def = policies.NormalTanhPolicy(
                action_dim=action_dim,
                hidden_dims=config.width_actor,
                depth=config.depth_actor,
                use_bronet=True,
            )
            critic_def = critic_net.DoubleCritic(
                hidden_dims=config.width_critic,
                depth=config.depth_critic,
                output_nodes=output_nodes,
                use_bronet=True,
            )
            actor = Model.create(
                actor_def,
                inputs=[actor_key, observations],
                tx=optax.adamw(learning_rate=config.actor_lr),
            )
            critic = Model.create(
                critic_def,
                inputs=[critic_key, observations, actions],
                tx=optax.adamw(learning_rate=config.critic_lr),
            )

            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
            if config.target_noise_kind == 'none':
                target_critic = ModelWrapper(target_critic)
            elif config.target_noise_kind == 'gauss':
                target_critic = GaussianNoiseModel(target_critic, sigma=config.target_noise)
            elif config.target_noise_kind == 'network':
                noise_mlp_def = MLPClassic(
                    hidden_dims=config.target_noise_network_width,
                    depth=config.target_noise_network_depth,
                    add_final_layer=True,
                    output_nodes=output_nodes,
                )
                noise_mlp_kw = dict(
                    model_def=noise_mlp_def,
                    inputs=[critic_key, observations, actions],
                    tx=optax.adamw(learning_rate=config.critic_lr),
                )
                noise_mlp1 = Model.create(**noise_mlp_kw)
                noise_mlp2 = Model.create(**noise_mlp_kw)
                target_critic = NetworkNoiseModel(
                    target_critic,
                    noise_net1=noise_mlp1,
                    noise_net2=noise_mlp2,
                    scale=config.target_noise,
                )

            temp = Model.create(
                temperature.Temperature(config.init_temperature),
                inputs=[temp_key],
                tx=optax.adam(learning_rate=config.temp_lr, b1=0.5),
            )

            if self.use_separate_critic:
                separate_critic_def = critic_net.DoubleCritic(
                    hidden_dims=config.separate_width_critic,
                    depth=config.separate_depth_critic,
                    output_nodes=output_nodes,
                    use_bronet=True,
                )
                separate_critic = Model.create(
                    separate_critic_def,
                    inputs=[critic_key, observations, actions],
                    tx=optax.adamw(learning_rate=config.critic_lr),
                )
            else:
                separate_critic_def = critic_net.NullDoubleCritic(output_nodes=output_nodes)
                separate_critic = Model.create(
                    separate_critic_def,
                    inputs=[critic_key, observations, actions],
                    tx=optax.adamw(learning_rate=config.critic_lr),
                )

            return actor, critic, target_critic, separate_critic, temp, rng

        self.init_models = jax.jit(jax.vmap(_init_models))
        self.reset()

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    # for compatibility
    def sample_actions_o(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def log_prob(
        self, observations: np.ndarray, actions: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        return policies.log_prob(
            self.actor.apply_fn, self.actor.params, observations, actions, temperature
        )

    def update(
        self, batch: Batch, num_updates: int, env_step: int, hard_target_update: bool = False
    ) -> InfoDict:
        step, rng, actor, critic, target_critic, separate_critic, temp, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.separate_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.distributional,
            self.quantile_taus,
            self.pessimism,
            self.use_hard_target_update,
            self.use_separate_critic,
            self.step,
            num_updates,
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.separate_critic = separate_critic
        self.temp = temp
        if hard_target_update:
            self.hard_target_update()

        return info

    def hard_target_update(self):
        self.target_critic = hard_target_update(self.critic, self.target_critic)

    def compute_temp(self):
        return _compute_temp_jit(self.temp)

    def compute_critic(self, observations, actions):
        return _compute_critic_jit(self.critic, observations, actions)

    def compute_target_critic(self, observations, actions):
        return _compute_target_critic_jit(self.target_critic, observations, actions, rng=self.rng)

    def compute_critic_batch(self, batch: Batch):
        return _compute_critic_batch_jit(self.critic, batch)

    def compute_separate_critic_batch(self, batch: Batch):
        return _compute_critic_batch_jit(self.separate_critic, batch)

    def compute_critic_loss(
        self, batch: Batch, critic: Optional[Model] = None, target_critic: Optional[Model] = None
    ):
        critic = critic or self.critic
        target_critic = target_critic or self.target_critic
        batch = jax.tree.map(lambda x: jnp.take(x, 0, axis=1), batch)
        if self.distributional:
            return _compute_quantile_loss_jit(
                self.rng,
                self.actor,
                critic,
                target_critic,
                self.temp,
                batch,
                self.discount,
                self.pessimism,
                self.quantile_taus,
            )
        else:
            return _compute_critic_loss_jit(
                self.rng,
                self.actor,
                critic,
                target_critic,
                self.temp,
                batch,
                self.discount,
                self.pessimism,
            )

    def compute_critic_grad_var(self, batch: Batch):
        batch = jax.tree.map(lambda x: jnp.take(x, 0, axis=1), batch)
        if self.distributional:
            return _compute_quantile_critic_grad_var_jit(
                self.rng,
                self.actor,
                self.critic,
                self.target_critic,
                self.temp,
                batch,
                self.discount,
                self.pessimism,
                self.quantile_taus,
            )
        else:
            return _compute_critic_grad_var_jit(
                self.rng,
                self.actor,
                self.critic,
                self.target_critic,
                self.temp,
                batch,
                self.discount,
                self.pessimism,
            )

    def _compute_critic_grad_norm(self, critic: Model, batch: Batch):
        batch = jax.tree.map(lambda x: jnp.take(x, 0, axis=1), batch)
        if self.distributional:
            return _compute_quantile_critic_grad_norm_jit(
                self.rng,
                self.actor,
                critic,
                self.target_critic,
                self.temp,
                batch,
                self.discount,
                self.pessimism,
                self.quantile_taus,
            )
        else:
            return _compute_critic_grad_norm_jit(
                self.rng,
                self.actor,
                critic,
                self.target_critic,
                self.temp,
                batch,
                self.discount,
                self.pessimism,
            )

    def compute_critic_grad_norm(self, batch: Batch):
        return self._compute_critic_grad_norm(self.critic, batch)

    def compute_separate_critic_grad_norm(self, batch: Batch):
        return self._compute_critic_grad_norm(self.separate_critic, batch)

    def get_adam_critic_grad_var(self):
        """Use Adam to estimate the gradient variance."""
        calc_variance = lambda nu, mu: nu - mu**2

        for state in self.critic.opt_state:
            if isinstance(state, optax._src.transform.ScaleByAdamState):
                grad_var = jax.tree_map(
                    calc_variance, state.nu, state.mu
                )  # Bias correction not needed since t is large

                all_sums = jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(
                        lambda x: jnp.sum(x, axis=tuple(range(1, x.ndim))), grad_var
                    )
                )
                all_counts = jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: jnp.prod(jnp.array(x.shape[1:])), grad_var)
                )
                total_sum = jnp.sum(jnp.stack(all_sums), axis=0)
                total_count = jnp.sum(jnp.stack(all_counts), axis=0)

                return {
                    'adam_mean_critic_grad_var': total_sum / total_count,
                    'adam_sum_critic_grad_var': total_sum,
                }

    def reset(self):
        self.actor, self.critic, self.target_critic, self.separate_critic, self.temp, self.rng = (
            self.init_models(self.seeds)
        )
        self.step = 1

    def save(self, path):
        self.actor.save(f'{path}/actor.txt')
        self.critic.save(f'{path}/critic.txt')
        self.target_critic.save(f'{path}/target_critic.txt')
        self.temp.save(f'{path}/temp.txt')

    def save_target_critic(self, path):
        self.target_critic.save(f'{path}/target_critic.txt')

    def load(self, path):
        self.actor = self.actor.load(f'{path}/actor.txt')
        self.critic = self.critic.load(f'{path}/critic.txt')
        self.target_critic = self.target_critic.load(f'{path}/target_critic.txt')
        self.temp = self.temp.load(f'{path}/temp.txt')

    def load_target_critic(self, path):
        self.target_critic = self.target_critic.load(f'{path}/target_critic.txt')

    def get_num_params(self):
        models = {
            'actor': self.actor,
            'critic': self.critic,
            'target_critic': self.target_critic,
            'separate_critic': self.separate_critic,
            'temp': self.temp,
        }
        return {
            name: sum(x.size for x in jax.tree.leaves(model.params)) // self.num_seeds
            for name, model in models.items()
        }


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='bro',
            actor_lr=3e-4,
            critic_lr=3e-4,
            temp_lr=3e-4,
            discount=0.99,
            tau=0.005,
            target_entropy=None,
            init_temperature=1.0,
            pessimism=0.0,
            num_seeds=5,
            updates_per_step=10,
            batch_size=256,
            distributional=True,
            n_quantiles=100,
            width_critic=512,
            depth_critic=2,
            width_actor=256,
            depth_actor=1,
            reset_interval=2500000,  # grad steps
            use_reset=True,
            hard_target_update=False,
            hard_target_update_interval=1000,  # grad steps
            target_noise_kind='none',  # 'none', 'gauss', 'network'
            target_noise=0.0,
            target_noise_network_width=256,
            target_noise_network_depth=2,
            use_separate_critic=False,  # for logging only
            separate_width_critic=512,
            separate_depth_critic=2,
        )
    )
    return config
