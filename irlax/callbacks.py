import gymnasium as gym
from flax import nnx

from utils import save_model, evaluate


def eval_callback(episode: int, model: nnx.Module, eval_env: gym.Env, evaluate_every: int = 1000,
    num_episodes: int = 10):
    if episode % evaluate_every == 0:
        mean_reward = evaluate(eval_env, model, num_episodes)
        print(f'Episode: {episode}, Mean Reward: {mean_reward}')


def save_model_callback(episode: int, model: nnx.Module, save_every: int = 1000, model_name: str = 'dqn_model',
                        chkp_dir: str = 'checkpoints'):
    if episode % save_every == 0:
        save_model(model, chkp_dir, model_name)
        print(f'Model saved at episode {episode}')
