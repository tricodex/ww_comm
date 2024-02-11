# eval_specific.py

from __future__ import annotations
import os
import supersuit as ss
from stable_baselines3 import PPO
from settings import env_kwargs
import waterworld_v4 

MODEL_DIR = 'models'
TRAIN_DIR = 'train'

def eval(env_fn, model_path, num_games=100, render_mode=None):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    if not os.path.exists(model_path):
        print("Model file not found at the specified path.")
        return

    model = PPO.load(model_path)
    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for a in env.agents:
                rewards[a] += env.rewards[a]
            
            if termination or truncation:
                break
            else:
                action = model.predict(obs, deterministic=True)[0]
            env.step(action)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards)
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")

    return avg_reward

if __name__ == "__main__":
    env_fn = waterworld_v4  
    model_path = input("models\\train\\waterworld_v4_20240118-235402.zip") #Enter the path of the PPO model to evaluate
    num_games = 3  # Number of games for evaluation
    render_mode = None  # Render mode, can be "human", None or ... not relevant

    # Run the evaluation
    eval(env_fn, model_path, num_games=num_games, render_mode=render_mode)
