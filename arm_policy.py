from torch import rand
import gym
import time
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from cassie import CassieRefEnv
from model_arm_imitation import *
import matplotlib.pyplot as plt
import random
import numpy as np
from math import floor

class ArmPolicy:
    def __init__(self,
                 model_path='arm_best_model',
                 datafile="dataset/data1.json",
                 render_mode='human',
                 learn_imitation=True):
        self.model = SAC.load(model_path)
        self.env = ArmEnv(datafile=datafile, render_mode=render_mode, learn_imitation=True )

    def get_action(self, ee_pose, ee_ori):
        sample = np.zeros(7)
        sample[0] = ee_pose[0]
        sample[1] = ee_pose[1]
        sample[2] = ee_pose[2]
        sample[3] = ee_ori[0]
        sample[4] = ee_ori[1]
        sample[5] = ee_ori[2]
        sample[6] = ee_ori[3]
        obs, _ = self.env.reset(target_pose=sample)

        action, _ = self.model.predict(obs, deterministic=True)
        return action