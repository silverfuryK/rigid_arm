import gym
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from model_wholebody_2 import *
import numpy as np
from typing import Callable

import wandb
from wandb.integration.sb3 import WandbCallback

register(
    id="CassieArmEnv-v0",
    entry_point="model_wholeboy:CassieArmEnv",
    max_episode_steps=600,
    autoreset=True,
)

render = None

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = CassieArmEnv(datafile="dataset/data1.json", render_mode=render)
        else:
            env = CassieArmEnv(datafile="dataset/data1.json", render_mode=render)
        return env
    return _f

def make_env_eval(env_id):
    def _f():
        if env_id == 0:
            env = CassieArmEnv(datafile="dataset/data1.json", render_mode=render)
        else:
            env = CassieArmEnv(datafile="dataset/data1.json", render_mode=render)
        return env
    return _f

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


class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, verbose=1):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:                

            # self.logger.record('reward/action', np.mean(self.training_env.get_attr('reward_action')))
            self.logger.record('reward/pelvis_pose', np.mean(self.training_env.get_attr('r_0')))
            self.logger.record('reward/pelvis_ori', np.mean(self.training_env.get_attr('r_1')))
            self.logger.record('reward/cassie_joint_pose', np.mean(self.training_env.get_attr('r_2')))
            self.logger.record('reward/cassie_torque', np.mean(self.training_env.get_attr('r_3')))
            self.logger.record('reward/cassie_max_action', np.mean(self.training_env.get_attr('r_4')))
            # self.logger.record('reward/cassie_footstep', np.mean(self.training_env.get_attr('')))
            self.logger.record('reward/ee_pose', np.mean(self.training_env.get_attr('reward_ee_pos')))
            self.logger.record('reward/ee_ori', np.mean(self.training_env.get_attr('reward_ee_ori')))
            self.logger.record('reward/cassie_qpos', np.mean(self.training_env.get_attr('reward_cassie_qpos')))
            self.logger.record('reward/del_ee_pos', np.mean(self.training_env.get_attr('del_ee_pos')))
            self.logger.record('reward/del_ee_ori', np.mean(self.training_env.get_attr('del_ee_pos')))

            self.logger.record('reward/totalreward', np.mean(self.training_env.get_attr('total_reward')))

            # self.logger.record('reward/ee', np.mean(self.training_env.get_attr('reward_arm')))    
            return True

if __name__ == '__main__':

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 4e7,
        "env_id": "CassieArmEnv-v0",
        "buffer_size": 100000,
        "train_freq": 5,
        "gradient_steps": 1,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": 0.01,
        "learning_rate": linear_schedule(5e-3),
        "n_envs": 21,
        "batch_size": 256,
        "seed": 1,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_demo/SAC/standing_arm_0",
    }

    load_trained_model = False
    num_envs = config["n_envs"]

    print("declaring envs")  
    train_envs =[make_env(i) for i in range(1, config["n_envs"])]
    train_env = VecMonitor(SubprocVecEnv(train_envs))

    eval_envs =[make_env(i) for i in range(1, 2)]
    eval_env = VecMonitor(SubprocVecEnv(eval_envs))

    # train_env = make_vec_env(
    #     config["env_id"], n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    # )
    # Separate evaluation env
    # eval_env = make_vec_env(config["env_id"], n_envs=1, vec_env_cls=SubprocVecEnv)

    # envs = Soft_Arm_Model_Imitation("cassie-softarm/softarm_2_OG.xml", ik_model_path = "ik_model/modelv3.h5",  datafile = "dataset/dataset_slider1.json", render=False)
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    
    run = wandb.init(
        project="Cassie_Arm_Imitation",
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
        verbose=0,
        log='all'
    )
    print("declaring eval env")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{run.name}/",
        log_path=f"./logs/{run.name}/",
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    tensorboard_callback = TensorboardCallback()
    
    callback_list = CallbackList([wandbcallback, eval_callback, tensorboard_callback])

    print("loading model")

    if load_trained_model:
        try:
            loadmodel = 'models_no_action_track'
            print("\n\n\n\n\nModel loaded\n\n\n\n\n")
            model = SAC.load(loadmodel, device='auto', env=train_env, tensorboard_log=f"logs/tensorboard/{run.name}/", verbose=0)
            print("\n\n\n\n\n\nModel loaded ready to go\n\n\n\n\n\n")
        except:
            loadmodel = None
            print("No model loaded, starting from scratch")
    else:

        model = SAC(
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
        
    model.is_tb_set = False
    print("Training started")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=config["progress_bar"],
    )
    model.save("./models_no_action_track")