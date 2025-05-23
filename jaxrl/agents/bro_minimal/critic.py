from typing import Tuple
import jax
import jax.numpy as jnp
from jaxrl.replay_buffer import Batch
from jaxrl.networks.common import InfoDict, Model, ModelWrapper, Params, PRNGKey, tree_norm


def _compute_loss_helper(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    critic_params: Params,
    target_critic: ModelWrapper,
    temp: Model,
    observations: jnp.ndarray,
    next_observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    discount: float,
    pessimism: float,
):
    """Compute from scratch"""
    key, target_critic_key = jax.random.split(key)
    dist = actor(next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(next_observations, next_actions, key=target_critic_key)
    next_q = (next_q1 + next_q2) / 2 - pessimism * jnp.abs(next_q1 - next_q2) / 2
    target_q = rewards + discount * masks * next_q
    target_q -= discount * temp() * masks * next_log_probs

    q1, q2 = critic.apply({'params': critic_params}, observations, actions)
    critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
    return critic_loss


def compute_loss(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
):
    """Compute from scratch"""
    return _compute_loss_helper(
        key,
        actor,
        critic,
        critic.params,
        target_critic,
        temp,
        batch.observations,
        batch.next_observations,
        batch.actions,
        batch.rewards,
        batch.masks,
        discount,
        pessimism,
    )


def _compute_quantile_loss_helper(
    key: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    quantile_critic_params: Params,
    target_quantile_critic: ModelWrapper,
    temp: Model,
    observations: jnp.ndarray,
    next_observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
):
    key, target_critic_key = jax.random.split(key)
    kappa = 1.0
    dist = actor(next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_quantile_critic(
        next_observations, next_actions, key=target_critic_key
    )
    next_q = (next_q1 + next_q2) / 2 - pessimism * jnp.abs(next_q1 - next_q2) / 2
    target_q = rewards[..., None, None] + discount * masks[..., None, None] * next_q[:, None, :]
    target_q -= discount * temp().mean() * masks[..., None, None] * next_log_probs[..., None, None]

    q1, q2 = quantile_critic.apply({'params': quantile_critic_params}, observations, actions)
    td_errors1 = target_q - q1[..., None]
    td_errors2 = target_q - q2[..., None]
    critic_loss = calculate_quantile_huber_loss(
        td_errors1, taus, kappa=kappa
    ) + calculate_quantile_huber_loss(td_errors2, taus, kappa=kappa)
    return critic_loss


def compute_quantile_loss(
    key: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
):
    return _compute_quantile_loss_helper(
        key,
        actor,
        quantile_critic,
        quantile_critic.params,
        target_quantile_critic,
        temp,
        batch.observations,
        batch.next_observations,
        batch.actions,
        batch.rewards,
        batch.masks,
        discount,
        pessimism,
        taus,
    )


def update(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
) -> Tuple[Model, InfoDict]:
    key, target_critic_key = jax.random.split(key)
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions, key=target_critic_key)
    next_q = (next_q1 + next_q2) / 2 - pessimism * jnp.abs(next_q1 - next_q2) / 2
    target_q = batch.rewards + discount * batch.masks * next_q
    target_q -= discount * temp() * batch.masks * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_fn = lambda actions: critic.apply(
            {'params': critic_params}, batch.observations, actions
        )

        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5 * (q1 + q2).mean(), (q1, q2)

        (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'r': batch.rewards.mean(),
            'critic_pnorm': tree_norm(critic_params),
            'critic_agnorm': jnp.sqrt((action_grad**2).sum(-1)).mean(0),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info['critic_gnorm'] = info.pop('grad_norm')
    return new_critic, info


def compute_critic_grad_var(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
):
    """Sample variance over examples in batch."""

    def single_example_loss_fn(params, example) -> jnp.ndarray:
        def add_batch_dim(x):
            return x[None, ...]

        obs = add_batch_dim(example.observations)
        next_obs = add_batch_dim(example.next_observations)
        actions = add_batch_dim(example.actions)
        rewards = add_batch_dim(example.rewards)
        masks = add_batch_dim(example.masks)

        return _compute_loss_helper(
            key,
            actor,
            critic,
            params,
            target_critic,
            temp,
            obs,
            next_obs,
            actions,
            rewards,
            masks,
            discount,
            pessimism,
        )

    def get_grad(example):
        return jax.grad(lambda params: single_example_loss_fn(params, example))(critic.params)

    batch_grads = jax.vmap(get_grad)(batch)
    grad_variance = jax.tree_map(lambda x: jnp.var(x, axis=0), batch_grads)
    flat_var, _ = jax.flatten_util.ravel_pytree(grad_variance)
    return {
        'mean_critic_grad_var': jnp.mean(flat_var),
        'sum_critic_grad_var': jnp.sum(flat_var),
    }


def compute_critic_grad_norm(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
) -> Tuple[Model, InfoDict]:
    def helper(critic_params):
        return _compute_loss_helper(
            key,
            actor,
            critic,
            critic_params,
            target_critic,
            temp,
            batch.observations,
            batch.next_observations,
            batch.actions,
            batch.rewards,
            batch.masks,
            discount,
            pessimism,
        )

    return jax.grad(helper)(critic.params)


def update_quantile(
    key: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: ModelWrapper,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
) -> Tuple[Model, InfoDict]:
    key, target_critic_key = jax.random.split(key)
    kappa = 1.0
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_quantile_critic(
        batch.next_observations, next_actions, key=target_critic_key
    )
    next_q = (next_q1 + next_q2) / 2 - pessimism * jnp.abs(next_q1 - next_q2) / 2
    target_q = (
        batch.rewards[..., None, None]
        + discount * batch.masks[..., None, None] * next_q[:, None, :]
    )
    target_q -= (
        discount * temp().mean() * batch.masks[..., None, None] * next_log_probs[..., None, None]
    )

    def critic_loss_fn(quantile_critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_fn = lambda actions: quantile_critic.apply(
            {'params': quantile_critic_params}, batch.observations, actions
        )

        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5 * (q1 + q2).mean(), (q1, q2)

        (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        td_errors1 = target_q - q1[..., None]
        td_errors2 = target_q - q2[..., None]
        critic_loss = calculate_quantile_huber_loss(
            td_errors1, taus, kappa=kappa
        ) + calculate_quantile_huber_loss(td_errors2, taus, kappa=kappa)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'r': batch.rewards.mean(),
            'critic_pnorm': tree_norm(quantile_critic_params),
            'critic_agnorm': jnp.sqrt((action_grad**2).sum(-1)).mean(0),
        }

    new_quantile_critic, info = quantile_critic.apply_gradient(critic_loss_fn)
    info['critic_gnorm'] = info.pop('grad_norm')
    return new_quantile_critic, info


def compute_quantile_critic_grad_var(
    key: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
):
    """Sample variance over examples in batch."""

    def single_example_loss_fn(params, example) -> jnp.ndarray:
        def add_batch_dim(x):
            return x[None, ...]

        obs = add_batch_dim(example.observations)
        next_obs = add_batch_dim(example.next_observations)
        actions = add_batch_dim(example.actions)
        rewards = add_batch_dim(example.rewards)
        masks = add_batch_dim(example.masks)

        return _compute_quantile_loss_helper(
            key,
            actor,
            quantile_critic,
            params,
            target_quantile_critic,
            temp,
            obs,
            next_obs,
            actions,
            rewards,
            masks,
            discount,
            pessimism,
            taus,
        )

    def get_grad(example):
        return jax.grad(lambda params: single_example_loss_fn(params, example))(
            quantile_critic.params
        )

    batch_grads = jax.vmap(get_grad)(batch)
    grad_variance = jax.tree_map(lambda x: jnp.var(x, axis=0), batch_grads)
    flat_var, _ = jax.flatten_util.ravel_pytree(grad_variance)
    return {
        'mean_critic_grad_var': jnp.mean(flat_var),
        'sum_critic_grad_var': jnp.sum(flat_var),
    }


def compute_quantile_critic_grad_norm(
    key: PRNGKey,
    actor: Model,
    quantile_critic: Model,
    target_quantile_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    pessimism: float,
    taus: jnp.ndarray,
):
    def helper(critic_params):
        return _compute_quantile_loss_helper(
            key,
            actor,
            quantile_critic,
            critic_params,
            target_quantile_critic,
            temp,
            batch.observations,
            batch.next_observations,
            batch.actions,
            batch.rewards,
            batch.masks,
            discount,
            pessimism,
            taus,
        )

    return tree_norm(jax.grad(helper)(quantile_critic.params))


def target_update(critic: Model, target_critic: ModelWrapper, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )
    return target_critic.replace_(params=new_target_params)


def hard_target_update(critic: Model, target_critic: ModelWrapper) -> Model:
    return target_critic.replace_(params=critic.params)


def huber_replace(td_errors, kappa: float = 1.0):
    return jnp.where(
        jnp.absolute(td_errors) <= kappa,
        0.5 * td_errors**2,
        kappa * (jnp.absolute(td_errors) - 0.5 * kappa),
    )


def calculate_quantile_huber_loss(td_errors, taus, kappa: float = 1.0):
    element_wise_huber_loss = huber_replace(td_errors, kappa)
    mask = jax.lax.stop_gradient(jnp.where(td_errors < 0, 1, 0))  # detach this
    element_wise_quantile_huber_loss = (
        jnp.absolute(taus[..., None] - mask) * element_wise_huber_loss / kappa
    )
    quantile_huber_loss = element_wise_quantile_huber_loss.sum(axis=1).mean()
    return quantile_huber_loss
