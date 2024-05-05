import os
import numpy as np
# from gymnasium import MujocoEnv
from gymnasium import utils
from gymnasium import *
import mujoco
import mujoco_viewer
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from typing import Dict

DEFAULT_CAMERA_CONFIG = {
	"trackbodyid": 1,
	"distance": 4.0,
	"lookat": np.array((0.0, 0.0, 2.0)),
	"elevation": -20.0,
}


def mass_center(model, data):
	mass = np.expand_dims(model.body_mass, axis=1)
	xpos = data.xipos
	return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class ArmEnv(Env):
	metadata = {
		"render_modes": [
			"human",
			"rgb_array",
			"depth_array",
		],
		"render_fps": 33,
	}

	def __init__(
		self,
		xml_string='xml/gen3.xml',
		forward_reward_weight=1.25,
		ctrl_cost_weight=0.1,
		healthy_reward=5.0,
		terminate_when_unhealthy=True,
		healthy_z_range=(0.8, 2.0),
		reset_noise_scale=1e-3,
		exclude_current_positions_from_observation=True,
		render_mode="human",
		datafile="dataset/data.json",
		ik_model_path = "ik_model/modelv1_256.h5",
		learn_imitation = False,
		play = False,
		**kwargs,
	):
		self._forward_reward_weight = forward_reward_weight
		self._ctrl_cost_weight = ctrl_cost_weight
		self._healthy_reward = healthy_reward
		self._terminate_when_unhealthy = terminate_when_unhealthy
		self._healthy_z_range = healthy_z_range

		self._reset_noise_scale = reset_noise_scale

		self._exclude_current_positions_from_observation = (
			exclude_current_positions_from_observation
		)

		self.observation_space = spaces.Dict(
			spaces={
				"observation":
				spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float64)
				},
		seed=None)

		# self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float64)

		self.action_space = spaces.Box(
			low=-1, high=1, shape=(6,), dtype=np.float64
		)

		self.frame_skip = 60

		self.xml_string = open(xml_string, 'r').read()

		self.model = mujoco.MjModel.from_xml_string(self.xml_string)
		self.data = mujoco.MjData(self.model)

		self.timestamp = 0
		self.max_timestamp = 100

		self.ee_vel_lim = 1e-2

		self.datafile = datafile
		self.play = play

		if datafile is not None and learn_imitation:
			self.datafile = datafile
			self.initialize_datasets()

		self._update_init_qpos()

		# print(f'final x pos {self.data.qpos[0]}, {self.ref_trajectory.qpos[0, 0]}')

		# self.sim = CassieSim(os.getcwd()+"/cassie.xml")
		# self.visual = True
		# if self.visual:
		#     self.vis = CassieVis(self.sim)
		# self.vis.draw(self.sim)
		# if render_mode:
		#     self.viewer = mujoco.MjViewer(self.sim)
		#     self.viewer_setup()

		self.render_mode = render_mode
		
		if self.render_mode == "human":
			self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
			self.viewer.render()
		# self.view = self.viewer_setup()
			

	def initialize_datasets(self):
		# Create a JSON file
		with open(self.datafile) as f:
			data = json.load(f)

		# df = pd.read_json(self.datafile)
		df = pd.json_normalize(data['data'])
		X = df.drop('actions', axis=1)
		y = pd.DataFrame(df['actions'].to_list())  # Assuming 'actions' is a list
		ee_position = pd.DataFrame(X['end-effector-position'].tolist(), columns=['ee_pos_x', 'ee_pos_y', 'ee_pos_z'])
		# ee_position = ee_position[['ee_pos_z', 'ee_pos_y', 'ee_pos_x']]

		X = pd.concat([X.drop('end-effector-position', axis=1), ee_position], axis=1)
		ee_orientation = pd.DataFrame(X['end-effector-orientation'].tolist(), columns=['ee_ori_w', 'ee_ori_x', 'ee_ori_y', 'ee_ori_z'])
		X = pd.concat([X.drop('end-effector-orientation', axis=1), ee_orientation], axis=1)
		# y_flat = y.apply(lambda x: pd.Series([i for _list in x for i in _list]), axis=1)

		# Split the data into training and testing sets
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		f.close()
	
	def write_observations_to_json(self,actions, obs, filename):
		# Create a dictionary to hold the data
		data = {
			"actions": actions,
			"end-effector-position": obs[12:15].tolist(),
			"end-effector-orientation": obs[15:19].tolist(),
			"shoulder-link-output": obs[6].tolist(),
			"bicep-link-output": obs[7].tolist(),
			"forearm-link-output": obs[8].tolist(),
			"sphericalWrist1-link-output": obs[9].tolist(),
			"sphericalWrist2-link-output": obs[10].tolist(),
			"bracelet-link-output": obs[11].tolist()
			
		}
		with open(filename, 'a') as f:
			json.dump(data, f)
			f.write(',\n')  # Write a newline character after each JSON object

		# print("observations written to file")
	
	def get_random_data(self, play = False, random = False):
		# sample random index from the dataset
		random_index = np.random.randint(0, len(self.X_train)-1)
		data_obs = np.array(self.X_train[random_index:random_index+1])
		data_action = np.array(self.y_train[random_index:random_index+1])

		if random:
			# data_obs[0][8] = data_obs[0][8] + np.random.uniform(-0.4, 0.2)
			data_obs[0][8] = 0.2
			data_obs[0][6:8] = data_obs[0][6:8] + np.random.uniform(-0.3, 0.3, 2)

		self.data_obs = np.reshape(data_obs, (data_obs.shape[1],))
		self.data_ee_target = self.data_obs[6:]
		self.data_action = np.reshape(data_action, (data_action.shape[1],))

	def _update_init_qpos(self):
		# handcrafted init qpos
		
		# self.set_state(self.init_qpos, self.init_qvel)
		mujoco.mj_resetData(self.model, self.data)
		# self.reset()


	def control_cost(self, action):
		control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
		return control_cost


	# def _get_obs(self):
	#     position = self.data.qpos.flat.copy()
	#     velocity = self.data.qvel.flat.copy()

	#     com_inertia = self.data.cinert.flat.copy()
	#     com_velocity = self.data.cvel.flat.copy()

	#     actuator_forces = self.data.qfrc_actuator.flat.copy()
	#     external_contact_forces = self.data.cfrc_ext.flat.copy()

	#     if self._exclude_current_positions_from_observation:
	#         position = position[2:]

	#     return np.concatenate(
	#         (
	#             position,
	#             velocity,
	#             com_inertia,
	#             com_velocity,
	#             actuator_forces,
	#             external_contact_forces,
	#         )
	#     )

	def _get_obs(self):

		

		position = self.data.qpos.flat.copy()
		velocity = self.data.qvel.flat.copy()

		com_inertia = self.data.cinert.flat.copy()
		com_velocity = self.data.cvel.flat.copy()

		actuator_forces = self.data.qfrc_actuator.flat.copy()
		external_contact_forces = self.data.cfrc_ext.flat.copy()

		# for shuldler, bicep, forearm, sphericalWrist1, sphericalWrist2, bracelet
		shoulder_link_input = self.data.sensor("shoulder-link-input").data
		shoulder_link_output = self.data.sensor("shoulder-link-output").data
		bicep_link_input = self.data.sensor("bicep-link-input").data
		bicep_link_output = self.data.sensor("bicep-link-output").data
		forearm_link_input = self.data.sensor("forearm-link-input").data
		forearm_link_output = self.data.sensor("forearm-link-output").data
		sphericalWrist1_link_input = self.data.sensor("sphericalWrist1-link-input").data
		sphericalWrist1_link_output = self.data.sensor("sphericalWrist1-link-output").data
		sphericalWrist2_link_input = self.data.sensor("sphericalWrist2-link-input").data
		sphericalWrist2_link_output = self.data.sensor("sphericalWrist2-link-output").data
		bracelet_link_input = self.data.sensor("bracelet-link-input").data
		bracelet_link_output = self.data.sensor("bracelet-link-output").data

		self.end_effector_pos = self.data.sensor("end-effector-pos").data
		self.end_effector_ori = self.data.sensor("end-effector-orientation").data


		self.observations= np.concatenate(
			(
				shoulder_link_input,
				bicep_link_input,
				forearm_link_input,
				sphericalWrist1_link_input,
				sphericalWrist2_link_input,
				bracelet_link_input,
				shoulder_link_output,
				bicep_link_output,
				forearm_link_output,
				sphericalWrist1_link_output,
				sphericalWrist2_link_output,
				bracelet_link_output,

				self.end_effector_pos,
				self.end_effector_ori,
				self.data_obs[6:]
			)
		)
		return dict(observation= self.observations.tolist())
	
	def do_simulation(self, ctrl, n_frames):
		for _ in range(n_frames):
			self.data.ctrl[:] = ctrl
			mujoco.mj_step(self.model, self.data)

	def step(self, action, target_pose = None, write_to_file = False):

		if target_pose is not None:

			self.data_obs[6:] = target_pose

		done = False

		action_vel = np.array([0,0,0,0,0,0])
		action = np.concatenate((action, action_vel))
		

		self.do_simulation(action, self.frame_skip)

		ee_vel = self.data.sensor("end-effector-vel").data

		ee_vel_norm = np.sqrt(np.sum(ee_vel**2))

		# print(f'ee_vel_norm: {ee_vel_norm:.6e}')

		action = action[0:6]

		observation_dict = self._get_obs()
		observation = self.observations

		
		
		# if self.render_mode == "human":
		#     self.render()
			
		if self.render_mode == "human":
			try:
				self.viewer.render()
			except:
				print("rendering failed, initialiing viewer again")
				# self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

				
		self.timestamp += 1
		# print("timestamp: ", self.timestamp)

		

		# define reward

		reward_weights = [0.5, 0.4, 0, 0.2]    
		
		reward = 0

		ee_target_pos = self.data_obs[6:9]
		ee_target_ori = self.data_obs[9:]
		# target_joint_pos = self.data_obs[0:6]

		actual_ee_pos = self.end_effector_pos
		actual_ee_ori = self.end_effector_ori

		target_action = self.data_action[0:6]
		actual_action = action[0:6]


		self.r1 = (-np.linalg.norm(ee_target_pos - actual_ee_pos))
		self.r2 = np.exp(-np.linalg.norm(ee_target_ori - actual_ee_ori))
		self.r3 = np.exp(-np.linalg.norm(target_action - actual_action))
		self.r4 = np.exp(-ee_vel_norm)

		reward =    (
					reward_weights[0]*self.r1 + 
					# reward_weights[1]*self.r2 + \
					# reward_weights[2]*self.r3 + \
					reward_weights[3]*self.r4 +
					0.1 * np.exp(-np.linalg.norm(action))
		)
		
		

		self.del_ee_pos = np.linalg.norm(ee_target_pos - actual_ee_pos)
		self.del_ee_ori = np.linalg.norm(ee_target_ori - actual_ee_ori)

		self.del_ee_pos = np.abs(np.linalg.norm(ee_target_pos- actual_ee_pos))

		if self.del_ee_pos < 0.001:
				print("del_ee_pos: ", self.del_ee_pos)
				print("pose reached within 1mm")

				reward += 1
		
		if (ee_vel_norm < self.ee_vel_lim and self.del_ee_pos < 1e-3 and self.del_ee_ori < 1e-3) or self.timestamp >= self.max_timestamp:
			# print("\n\nwriting to file\n\n")
			# print(" action: ",action.copy())
			

			

			if write_to_file:
				self.write_observations_to_json(action, observation, self.datafile)
			# self.write_observations_to_json(action, observation, self.datafile)
			# self.reset_model()
			done = True

		else:
			done = False

		# if self.timestamp >= 500:
		#     done = True
		self.total_reward += reward

		terminated = done

		# define info

		self.info = {
			"reward_ee_pos": self.r1,
			"reward_ee_ori": self.r2,
			"reward_action": self.r3,
			"reward_ee_vel": self.r4,
			"del_ee_pos": self.del_ee_pos,
			"del_ee_ori": self.del_ee_ori,
			"total_reward": self.total_reward
			
		}

		return observation_dict, reward, terminated, False, self.info

	def reset_model(self, target_pose = None, random_sample = False, play = False):
		if not play:
			if random_sample:
				reset_bool = np.random.choice([True, False])
				if reset_bool:
					self._update_init_qpos()
			else:
				self._update_init_qpos()

		else:
			self._update_init_qpos()
		
		self.timestamp = 0
		self.total_reward = 0

		if target_pose is None:
			self.get_random_data(play = self.play, random=True)
		else:
			self.get_random_data()
			self.data_obs[6:9] = target_pose

		observation_dict = self._get_obs()
		observation = self.observations

		self.info = {
			"reward_ee_pos": 0,
			"reward_ee_ori": 0,
			"reward_action": 0,
			"reward_ee_vel": 0,
			"del_ee_pos": None,
			"del_ee_ori": None,
			"total_reward": self.total_reward
			
		}

		return observation_dict, self.info
	
	def reset(self, target_pose = None, seed = None, play = False):
		target_pose = None
		obs, info = self.reset_model(target_pose = target_pose, random_sample=False, play = play)
		return obs, info
	
 
	def run_action_data_collection(self, action = None, pred_target_render = False):
		
		done = False
		vel_pose = np.array([0,0,0,0,0,0])
		action = np.concatenate((action, vel_pose))
		# print("action",self.action_pos.copy())
		action = action

		# if pred_target_render:
			# self.model.body('target').pos = np.reshape(pred_target_render, (3,))
			# self.model.body('target').quat = self.target_ori
		while not done:
			obs, _, _, done, _ = self.step(action)

		# get ee pose
		obs = obs[0]["observation"]
		ee_pos = obs[12:15]
		ee_ori = obs[15:19]
		return ee_pos, ee_ori

	def viewer_setup(self):
		assert self.viewer is not None
		for key, value in DEFAULT_CAMERA_CONFIG.items():
			if isinstance(value, np.ndarray):
				getattr(self.viewer.cam, key)[:] = value
			else:
				setattr(self.viewer.cam, key, value)
