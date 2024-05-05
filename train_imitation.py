import gymnasium as gym
import sys
from typing import Callable
import datetime
import time
import optuna
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac import SAC
from stable_baselines3.common.env_checker import check_env
# from roboticsgym.algorithms.sb3.ipmd import IPMD

import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.envs.registration import register

from model_arm_imitation import ArmEnv

register(
    id="ArmEnv-v0",
    entry_point="model_arm_imitation:ArmEnv",
    max_episode_steps=600,
    autoreset=True,
)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = ArmEnv(datafile="dataset/data1.json", render_mode=None)
        else:
            env = ArmEnv(datafile="dataset/data1.json", render_mode=None)
        return env
    return _f

def train_expert_policy():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1e7,
        "env_id": "ArmEnv-v0",
        "buffer_size": 100000,
        "train_freq": 5,
        "gradient_steps": 1,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": 0.01,
        "learning_rate": linear_schedule(5e-3),
        "n_envs": 3,
        "batch_size": 256,
        "seed": 1,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_demo/SAC/buffer10trajArm",
    }
    run = wandb.init(
        project="Arm_Imitation",
        config=config,
        name=f'{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    wandbcallback = WandbCallback(
        model_save_path=f"models/{run.id}",
        model_save_freq=2000,
        gradient_save_freq=2000,
        verbose=1,
    )
    # Create log dir
    # train_env = make_vec_env(
    #     config["env_id"], n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    # )
    # Separate evaluation env
    num_envs = config["n_envs"]

    # envs =[make_env(i) for i in range(1, num_envs)]
    # train_env = VecMonitor(SubprocVecEnv(envs))

    # e_envs =[make_env(i) for i in range(1, num_envs)]
    # eval_env = VecMonitor(SubprocVecEnv(e_envs))

    train_env = make_vec_env(
        config["env_id"], n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    )
    # Separate evaluation env
    eval_env = make_vec_env(config["env_id"], n_envs=1, vec_env_cls=SubprocVecEnv)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{run.name}/",
        log_path=f"./logs/{run.name}/",
        eval_freq=2000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    callback_list = CallbackList([eval_callback, wandbcallback])
    # Init model
    irl_model = SAC(
        "MultiInputPolicy",
        env=train_env,
        gamma=0.99,
        verbose=config["verbose"],
        buffer_size=config["buffer_size"],
        ent_coef=config["ent_coef"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_steps=config["gradient_steps"],
        learning_starts=100,
        tensorboard_log=f"logs/tensorboard/{run.name}/",
    )
    # Model learning
    irl_model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=config["progress_bar"],
    )

    return f"logs/{run.name}/best_model"

train_expert_policy()