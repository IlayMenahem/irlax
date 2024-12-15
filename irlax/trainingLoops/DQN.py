'''
This module contains the training loop for a DQN
'''
from typing import Callable
from flax import nnx
import jax
import jax.numpy as jnp
from chex import Numeric
import rlax
from tqdm import tqdm
import optax
import gymnasium as gym

from irlax.replayBuffers import ReplayBuffer


def get_epsilon_greedy_action(model: nnx.Module, obs: jnp.ndarray, epsilon: Numeric) -> int:
    qVals = model(obs)
    a = rlax.epsilon_greedy(epsilon).sample(qVals)
    a = int(a[0])

    return a


def loss_fn(model: nnx.Module, obs_tm1: list, a_tm1: jnp.ndarray, r_t: jnp.ndarray, discount_t: float,
    obs_t: list) -> jnp.ndarray:
    model.train()

    q_tm1 = model(obs_tm1)
    q_t = model(obs_t)

    td_error = jax.vmap(rlax.q_learning)(q_tm1, a_tm1, r_t, discount_t, q_t)
    loss = jnp.mean(rlax.l2_loss(td_error))

    return loss


def learner_step(model: nnx.Module, optimizer: nnx.Optimizer, batch) -> jnp.ndarray:
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grad = grad_fn(model, *batch)
    optimizer.update(grad)

    return loss


def run_episode(env: gym.Env, model: nnx.Module, replay_buffer: ReplayBuffer, epsilon_by_frame: optax.Schedule,
    nnx_optimizer: nnx.Optimizer, episode: int, step: int, target_period: int, discount_factor: float = 0.99,
    step_limit: int = 50) -> tuple[float, int]:

    obs_prev, _ = env.reset()
    episodic_reward = 0.0

    for _ in range(step_limit):
        action = get_epsilon_greedy_action(model, obs_prev, epsilon_by_frame(episode))
        obs, reward, done, truncated, _ = env.step(action)
        episodic_reward = float(reward) + episodic_reward * discount_factor
        step += 1

        replay_buffer.push((obs_prev, action, episodic_reward, discount_factor, obs))
        obs_prev = obs

        if replay_buffer.is_ready() and step % target_period == 0:
            batch = replay_buffer.sample()
            _ = learner_step(model, nnx_optimizer, batch)

        if done or truncated:
            break

    return episodic_reward, step


def train_loop(env: gym.Env, model: nnx.Module, replay_buffer, optimizer: nnx.Optimizer, epsilon_by_frame:optax.Schedule,
    episode_count: int = 10000, target_period: int = 128, callbacks: list[Callable] = []) -> nnx.Module:

    '''
    Training loop for a DQN model
    env: gym.Env: The environment to train on
    model: nnx.Module: The model to train, gets observation as input and returns Q-values
    replay_buffer: ReplayBuffer: The replay buffer to store experiences
    optimizer: nnx.Optimizer: The optimizer to use
    epsilon_by_frame: optax.Schedule: The schedule for epsilon decay
    episode_count: int: The number of episodes to train for
    target_period: int: The number of steps between target network updates
    callbacks: list[callable]: List of callbacks to run after each episode
    '''
    step = 0
    progress_bar = tqdm(range(episode_count))
    avg_reward = nnx.metrics.Average('reward')

    for episode in range(episode_count):
        episodic_reward, step = run_episode(env, model, replay_buffer, epsilon_by_frame, optimizer, episode,
                              step, target_period)
        progress_bar.update(1)
        avg_reward.update(reward=episodic_reward)

        for callback in callbacks:
            callback(episode, model)

    return model
