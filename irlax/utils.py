import os
import orbax.checkpoint as ocp
from flax import nnx
import jax
import jax.numpy as jnp
import rlax
import gymnasium as gym

# inference functions
def dqn_action(model: nnx.Module, obs: jnp.ndarray) -> int:
    model.eval()

    qVals = model(obs)
    a = rlax.greedy().sample(qVals)
    a = int(a[0])

    return a


# Evaluation functions
def evaluate_episode(env: gym.Env, model: nnx.Module, step_limit: int = 50) -> float:
    obs, _ = env.reset()
    episode_reward: float = 0.0

    for _ in range(step_limit):
        action = dqn_action(model, obs)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += float(reward)

        if done or truncated:
            break

    return episode_reward


def evaluate(env: gym.Env, model: nnx.Module, num_episodes: int, step_limit: int = 50) -> jax.Array:
    model.eval()
    rewards = jnp.array([evaluate_episode(env, model, step_limit) for _ in range(num_episodes)])
    mean_reward = jnp.mean(rewards)

    return mean_reward


def save_model(model: nnx.Module, chkp_dir: str, model_name: str) -> None:
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)
    model_path = os.path.join(os.path.abspath(chkp_dir), model_name)

    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(model_path, state, force=True)


def load_model(model: nnx.Module, chkp_dir: str, model_name: str) -> nnx.Module:
    model_path = os.path.join(os.path.abspath(chkp_dir), model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found')

    graphdf, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    resotred_state = checkpointer.restore(model_path, state)

    model = nnx.merge(graphdf, resotred_state)

    return model
