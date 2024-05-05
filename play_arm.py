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


if __name__ == '__main__':
    t = time.monotonic()
    # model = PPO.load("model_saved/ppo_cassie_"+str(512 * 36)+"00")
    # model = PPO.load("./wrenchflatmodel/ppo_cassie-2969600")
    # model = PPO.load("standstill")

    model = SAC.load('best_model_arm_new copy')
    
    # model = RecurrentPPO.load('models_lstm_no_action_track')
    # cassie = CassieRefEnv(dynamics_randomization=False)
    # env = Soft_Arm_Model("/home/fury/LIDAR/Soft-Arm-Mujoco/cassie-softarm/softarm_2_OG.xml", render=True)
    env = ArmEnv(datafile="dataset/data1.json", render_mode='human', learn_imitation=True)
    # env = Soft_Arm_Model_Imitation("cassie-softarm/cassie_soft_arm_v2.xml", ik_model_path = "ik_model/modelv3.h5",  datafile = "dataset/dataset_slider1.json", render=True)
    obs, _ = env.reset()


    print("shape", len(obs["observation"]))
    # print(len(obs))
    action_arr = []
    ee_pose_arr = []
    targ_pose_arr = []    
    # trace circle
    radius = 0.3
    time_factor = 0.05
    
    start_time = time.monotonic()

    datafile = "dataset/data1.json"

    with open(datafile) as f:
            data = json.load(f)
            f.close()
            
    df = pd.json_normalize(data['data'])
    X = df.drop('actions', axis=1)
    y = pd.DataFrame(df['actions'].to_list())  # Assuming 'actions' is a list
    ee_position = pd.DataFrame(X['end-effector-position'].tolist(), columns=['ee_pos_x', 'ee_pos_y', 'ee_pos_z'])
    X = pd.concat([X.drop('end-effector-position', axis=1), ee_position], axis=1)
    ee_orientation = pd.DataFrame(X['end-effector-orientation'].tolist(), columns=['ee_ori_w', 'ee_ori_x', 'ee_ori_y', 'ee_ori_z'])
    X = pd.concat([X.drop('end-effector-orientation', axis=1), ee_orientation], axis=1)

    # make different datadrame for ee position and orientation
    X_ee = X[['ee_pos_x', 'ee_pos_y', 'ee_pos_z', 'ee_ori_w', 'ee_ori_x', 'ee_ori_y', 'ee_ori_z']]
    
    # sample from the dataset
    sample = X_ee.sample()
    sample = sample.to_numpy()[0]

    targ_x = radius * np.cos(t*time_factor)
    targ_y = radius * np.sin(t*time_factor)
    targ_z = 1.2

    sample[0] = targ_x
    sample[1] = -targ_y
    sample[2] = targ_z
    sample[3] = 0
    sample[4] = -0.8509035
    sample[5] = 0
    sample[6] = 0.525322
    
    # print("sample", sample.to_numpy())
    terminated = False
    env.reset(target_pose=sample, play=True)

    i = 0
    
    while True:

        ob = obs["observation"]
        

        # print("obs shape", len(ob))
        targ_x = radius * np.cos(t*time_factor)
        targ_y = radius * np.sin(t*time_factor)
        targ_z = 1.2

        sample[0] = targ_x
        sample[1] = -targ_y
        sample[2] = targ_z
        sample[3] = 0
        sample[4] = -0.8509035
        sample[5] = 0
        sample[6] = 0.525322
        # print("sample", sample)

        # env.data_obs[6:] = sample

        # print("shape", len(obs["observation"]))
        # target_pos = obs[0:3]
        # print("target_pose", sample[0:3])

        action, _states = model.predict(obs,deterministic=True)

        # print("action", action)
        # action, _ = model.predict(obs,deterministic=True)
        obs, rewards, terminated, _, info = env.step(action)
        # print(len(cassie.sim.qpos()))
        # while time.monotonic() - t < 60*0.0005:
        #     time.sleep(0.0001)
        #     # cassie.render()
        t = time.monotonic() - start_time
        # print("time", floor(t))
        
        # pos_index = np.array([7, 8, 9, 14, 20, 21, 22, 23, 28, 34])
        # qpos=np.array(cassie.sim.qpos())
        # joints=qpos[pos_index].tolist()
        # vel.append(joints)
        
        # vel.append(cassie.sim.qvel()[0])

        action_arr.append(action)
        ee_pose_arr.append(ob[9:12])
        targ_pose_arr.append(sample.tolist())

        i += 1
        if i > 100:
            
            print(np.linalg.norm(obs["observation"][12:15] - sample[0:3]))
            
            i = 0
            # targ_x = radius * np.cos(t*time_factor)
            # targ_y = radius * np.sin(t*time_factor)
            # targ_z = 1.2

            # sample[0] = targ_x
            # sample[1] = -targ_y
            # sample[2] = targ_z
            # sample[3] = 0
            # sample[4] = -0.8509035
            # sample[5] = 0
            # sample[6] = 0.525322
            env.reset()
            print("r")


        # if floor(t) % 10 == 0:
        #     #save data in numpy array
        #     # action_arr = np.array(action_arr)
        #     # ee_pose_arr = np.array(ee_pose_arr)
        #     # targ_pose_arr = np.array(targ_pose_arr)
        #     # np.save('action_arr.npy', action_arr)
        #     # np.save('ee_pose_arr.npy', ee_pose_arr)
        #     # np.save('targ_pose_arr.npy', targ_pose_arr)
        #     sample = X_ee.sample()
        #     sample = sample.to_numpy()[0]

        #     env.reset(target_pose=sample)

        #     terminated = False
            # break

        # if terminated: #cassie.qpos[2]<0.6:
        #     # plt.plot(vel)
        #     # plt.show()
        #     # plt.legend()

        #     vel = []
        #     # cassie.setforce = 0# random.uniform(0,8)#0-8
        #     # print("yforce:",cassie.setforce)
        #     obs, _ = env.reset()

        #     # print(cassie.speed)
            
