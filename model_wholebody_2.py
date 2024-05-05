import os
from shlex import join
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium import *
import mujoco
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
import mujoco_viewer
from arm_policy import ArmPolicy

# from cassie_m.cassiemujoco import CassieSim, CassieVis

from reference_trajectories.loadstep import CassieTrajectory


DEFAULT_CAMERA_CONFIG = {
	"trackbodyid": 1,
	"distance": 4.0,
	"lookat": np.array((0.0, 0.0, 2.0)),
	"elevation": -20.0,
}

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def point_in_polygon(point, polygon):
    # Convert point and polygon to NumPy arrays for vectorized operations
    x, y = point
    polygon = np.array(polygon)
    

    x_vertices = polygon[:, 0]
    y_vertices = polygon[:, 1]
    

    x_vertices_shifted = np.roll(x_vertices, -1)
    y_vertices_shifted = np.roll(y_vertices, -1)
    

    cond_min_y = y > np.minimum(y_vertices, y_vertices_shifted)
    cond_max_y = y <= np.maximum(y_vertices, y_vertices_shifted)
    cond_max_x = x <= np.maximum(x_vertices, x_vertices_shifted)
    

    x_intersections = ((y - y_vertices) * (x_vertices_shifted - x_vertices) / 
                       (y_vertices_shifted - y_vertices + np.finfo(float).eps)) + x_vertices
    

    valid_edges = cond_min_y & cond_max_y & cond_max_x
    intersections = x <= x_intersections
    valid_intersections = valid_edges & intersections

    inside = valid_intersections.sum() % 2 == 1
    
    return inside




class CassieArmEnv(Env):
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
		xml_string='xml/cassie_w_gen3.xml',
		forward_reward_weight=1.25,
		ctrl_cost_weight=0.1,
		healthy_reward=5.0,
		terminate_when_unhealthy=True,
		healthy_z_range=(0.5, 2.0),
		reset_noise_scale=1e-5,
		exclude_current_positions_from_observation=True,
		arm_policy_model_path="best_model_arm_new copy",
		datafile="dataset/data1.json",
		render_mode = "human",
		max_episode_steps=600,
		**kwargs,
	):
		self._forward_reward_weight = forward_reward_weight
		self._ctrl_cost_weight = ctrl_cost_weight
		self._healthy_reward = healthy_reward
		self._terminate_when_unhealthy = terminate_when_unhealthy
		self._healthy_z_range = healthy_z_range

		self.pelvis_z_offset = - 0

		self._reset_noise_scale = reset_noise_scale

		self._exclude_current_positions_from_observation = (
			exclude_current_positions_from_observation
		)

		if exclude_current_positions_from_observation:
			observation_space = spaces.Box(
				low=-np.inf, high=np.inf, shape=(688,), dtype=np.float64
			)
		else:
			observation_space = spaces.Box(
				low=-np.inf, high=np.inf, shape=(688,), dtype=np.float64
			)
		# spaces.Dict()
		self.observation_space = spaces.Dict(
			spaces={
				"observation":
				spaces.Box(low=-np.inf, high=np.inf, shape=(115,), dtype=np.float64)
				},
		seed=None)

		action_bounds = np.array([4.5, 4.5, 12.2, 12.2, 0.9,4.5, 4.5, 12.2, 12.2, 0.9, 3.14159, 2.40855, 2.659881, 3.14159, 2.23402, 3.14159])

		self.action_space = spaces.Box(
			low=-action_bounds, high=action_bounds, shape=(16,), dtype=np.float64
		)

		self.frame_skip = 5
		self.xml_string = open(xml_string, 'r').read()

		self.model = mujoco.MjModel.from_xml_string(self.xml_string)
		self.data = mujoco.MjData(self.model)

		self.ref_trajectory = CassieTrajectory(
			os.getcwd() + "/reference_trajectories/cassie_walk/cassie_walking_from_stand.mat"
		)

		self.datafile = datafile

		self.initialize_datasets()
		self.get_random_data()

		self.arm_policy = ArmPolicy(model_path=arm_policy_model_path, datafile=datafile, render_mode=None)

		self.timestamp = 0
		self.max_episode_steps = max_episode_steps

		self._update_init_qpos()

		self.render_mode = render_mode

		# self.render = None

		if self.render_mode == "human":
			self.mj_viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
			# self.mj_viewer.render()
		else:
			self.mj_viewer = None

		# print(f'final x pos {self.data.qpos[0]}, {self.ref_trajectory.qpos[0, 0]}')

		# self.sim = CassieSim(os.getcwd()+"/cassie.xml")
		# self.visual = True
		# if self.visual:
		#     self.vis = CassieVis(self.sim)
		# self.vis.draw(self.sim)


		# render mj_viewer
		# self.render = True
		# if self.render:
		# 	self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
		# 	self.viewer.render()

	def initialize_datasets(self):
		# Create a JSON file
		with open(self.datafile) as f:
			data = json.load(f)

		# df = pd.read_json(self.datafile)
		df = pd.json_normalize(data['data'])
		X = df.drop('actions', axis=1)
		y = pd.DataFrame(df['actions'].to_list())  # Assuming 'actions' is a list
		ee_position = pd.DataFrame(X['end-effector-position'].tolist(), columns=['ee_pos_x', 'ee_pos_y', 'ee_pos_z'])
		# swap z and x columns
		# ee_position = ee_position[['ee_pos_z', 'ee_pos_y', 'ee_pos_x']]
		X = pd.concat([X.drop('end-effector-position', axis=1), ee_position], axis=1)
		ee_orientation = pd.DataFrame(X['end-effector-orientation'].tolist(), columns=['ee_ori_w', 'ee_ori_x', 'ee_ori_y', 'ee_ori_z'])
		X = pd.concat([X.drop('end-effector-orientation', axis=1), ee_orientation], axis=1)
		# y_flat = y.apply(lambda x: pd.Series([i for _list in x for i in _list]), axis=1)

		# Split the data into training and testing sets
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		f.close()

	def get_random_data(self, play = False, random = False):
		# sample random index from the dataset
		random_index = np.random.randint(0, len(self.X_train)-1)
		data_obs = np.array(self.X_train[random_index:random_index+1])
		data_action = np.array(self.y_train[random_index:random_index+1])

		self.data_obs = np.reshape(data_obs, (data_obs.shape[1],))
		self.data_ee_target = self.data_obs[6:]
		self.data_action = np.reshape(data_action, (data_action.shape[1],))

		if random:
			self.data_obs[8] = self.data_obs[8] + np.random.uniform(-0.4, 0)
			self.data_ee_target = self.data_obs[6:]
			
	def get_cassie_joint_pos(self):
		j0 = self.data.sensor("left-hip-roll-input").data
		j1 = self.data.sensor("left-hip-yaw-input").data
		j2 = self.data.sensor("left-hip-pitch-input").data
		j3 = self.data.sensor("left-knee-input").data
		j4 = self.data.sensor("left-foot-input").data
		j5 = self.data.sensor("right-hip-roll-input").data
		j6 = self.data.sensor("right-hip-yaw-input").data
		j7 = self.data.sensor("right-hip-pitch-input").data
		j8 = self.data.sensor("right-knee-input").data
		j9 = self.data.sensor("right-foot-input").data
		joint_pos = np.array([j0, j1, j2, j3, j4, j5, j6, j7, j8, j9])
		return joint_pos

	def set_cassie_qpos(self, qpos):
		# when arm is declared above legs
		self.data.qpos[0:35] = qpos
		return self.data.qpos.copy()
	
	def get_cassie_qpos(self, qpos):
		# floor_idx_offset = 0
		# pelvis_pos = qpos[:7]
		# pelvis_pos[2] = pelvis_pos[2] + self.pelvis_z_offset
		# body_pos = qpos[-28:]
		# # env.data.qpos[floor_idx_offset:7+floor_idx_offset] = pelvis_pos
		data_qpos = self.data.qpos.copy()
		# data_qpos[floor_idx_offset:7+floor_idx_offset] = pelvis_pos
		cassie_qpos = data_qpos[:35]
		return cassie_qpos
	
	def set_qvel(self, qvel):
		floor_idx_offset = 0
		pelvis_vel = qvel[:5]
		body_vel = qvel[-26:]
		# env.data.qpos[floor_idx_offset:7+floor_idx_offset] = pelvis_pos
		data_qpos = self.data.qvel.copy()
		data_qpos[floor_idx_offset:5+floor_idx_offset] = pelvis_vel
		data_qpos[-26:] = body_vel
		return data_qpos
	
	def get_cassie_qvel(self, qvel):
		data_qpos = self.data.qvel.copy()
		cassie_qvel = data_qpos[:32]
		return data_qpos
	
	def select_cassie_qpos(self, qpos):
		floor_idx_offset = 0
		pelvis_pos = qpos[:7]
		body_pos = qpos[-28:]
		# env.data.qpos[floor_idx_offset:7+floor_idx_offset] = pelvis_pos
		data_qpos = np.concatenate((pelvis_pos, body_pos))
		return data_qpos
	
	def select_cassie_qvel(self, qvel):
		floor_idx_offset = 0
		pelvis_vel = qvel[:5]
		body_vel = qvel[-26:]
		# env.data.qpos[floor_idx_offset:7+floor_idx_offset] = pelvis_pos
		data_qvel = np.concatenate((pelvis_vel, body_vel))
		return data_qvel
	
	def pelvis_center(self):
		pelvis_pos = self.data.sensor("pelvis-pose").data
		pelvis_ori = self.data.sensor("pelvis-orientation").data
		pelvis_vel = self.data.sensor("pelvis-vel").data
		# mass = np.expand_dims(model.body_mass, axis=1)
		# xpos = data.xipos
		return pelvis_pos.copy(), pelvis_ori.copy(), pelvis_vel.copy()
	
	def com_center_pos(self):
		mass = np.expand_dims(self.model.body_mass, axis=1)
		xpos = self.data.xipos
		return (np.sum(mass * xpos, axis=0) / np.sum(mass)).copy()

	def compute_relative_pose(self, ee_ori = None, ee_pos=None):
		b_mat = self.data.geom_xmat[self.model.geom('arm-base-geom').id]
		b_pos = self.data.geom_xpos[self.model.geom('arm-base-geom').id]
		# b_pos = env.data.sensor("base-link-pos").data
		# b_pos = env.data.sensor("base-link-pos").data
		if ee_ori is None:
			ee_mat = self.data.geom_xmat[self.model.geom('ee').id]
		else:
			ee_mat = quaternion_to_rotation_matrix(ee_ori)

		if ee_pos is None:
			ee_pos = self.data.geom_xpos[self.model.geom('ee').id]

		# ee_pos = self.data.geom_xpos[self.model.geom('ee').id]
		mat = compute_relative_transformation(b_mat, b_pos, ee_mat, ee_pos)

		tm_ori, tm_pos = recover_pose(mat)
		return tm_ori, tm_pos

	def _update_init_qpos(self):
		# handcrafted init qpos
		mujoco.mj_resetData(self.model, self.data)
		foot_offset = -0.75
		self.init_qpos_reset = np.array(
			[
				0,
				0,
				1.01,
				1,
				0,
				0,
				0,
				0.0045,
				0,
				0.4973,
				0.9785,
				-0.0164,
				0.01787,
				-0.2049,
				-1.1997,
				0,
				1.4267,
				0,
				-1.5244,
				1.5244,
				-1.5968 + foot_offset,
				-0.0045,
				0,
				0.4973,
				0.9786,
				0.00386,
				-0.01524,
				-0.2051,
				-1.1997,
				0,
				1.4267,
				0,
				-1.5244,
				1.5244,
				-1.5968 + foot_offset,
			]
		)

		# ref_qpos_cassie, ref_qvel_cassie = self.ref_trajectory.state(170)

		self.init_qpos = self.set_cassie_qpos(self.init_qpos_reset)

		# self.init_qpos = self.set_cassie_qpos(ref_qpos_cassie)

		# self.init_qpos = self.ref_trajectory.qpos[0]

		self.do_simulation(np.zeros(22,), 1)
		
		# self.init_qpos_reset = np.array(
		# 	[
		# 		0.04529737116916673,
		# 		-0.15300356752917388,
		# 		0.9710095501646747,
		# 		1.0,
		# 		0.0,
		# 		0.0,
		# 		0.0,
		# 		0.04516039439535328,
		# 		0.0007669903939428207,
		# 		0.48967542963286953,
		# 		0.5366660119008494,
		# 		-0.5459706642036749,
		# 		0.13716932320803393,
		# 		0.6285620114754674,
		# 		-1.3017744461194523,
		# 		-0.03886484136807136,
		# 		1.606101853366077,
		# 		-0.7079960941663008,
		# 		-1.786147490968169,
		# 		0.3175519006511133,
		# 		-1.683487349162938,
		# 		-0.04519107401111099,
		# 		-0.0007669903939428207,
		# 		0.4898192403317338,
		# 		0.38803053590372555,
		# 		-0.25971548696569596,
		# 		0.49875340077344466,
		# 		-0.7302227155144948,
		# 		-1.3018703199186952,
		# 		-0.038780951793733864,
		# 		1.606065900691361,
		# 		0.49858954295641644,
		# 		-1.6206843700546072,
		# 		0.12408356187240471,
		# 		-1.6835283352121144,
		# 	]
		# )

		# self.init_qvel = self.set_qvel(self.ref_trajectory.qvel[0])
		
		# self.set_state(self.init_qpos, self.init_qvel)
		# self.reset()

	def get_observations(self, print_data=False):
		qpos = self.data.qpos
		qvel = self.data.qvel
		# observations_full = self.get_observations_full(print_data)

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

		end_effector_pos = self.data.sensor("end-effector-pos").data
		end_effector_ori = self.data.sensor("end-effector-orientation").data

		base_pos = self.data.sensor("base-link-pos").data
		base_ori = self.data.sensor("base-link-ori").data
		self.end_effector_ori, self.end_effector_pos = self.compute_relative_pose(end_effector_ori, end_effector_pos)


	
		# cassie_joint_data = self.data.qpos[-28:]
		cassie_pelvis_pose = self.data.sensor("pelvis-pose").data
		cassie_pelvis_vel = self.data.sensor("pelvis-vel").data
		cassie_pelvis_orientation = self.data.sensor("pelvis-orientation").data
		cassie_pelvis_ang_vel = self.data.sensor("pelvis-angular-velocity").data
		cassie_pelvis_linear_acc = self.data.sensor("pelvis-linear-acceleration").data

		# cassie_left_foot_pose = self.data.sensor("left-foot-pose").data
		# cassie_left_foot_vel = self.data.sensor("left-foot-vel").data
		# cassie_left_foot_acc = self.data.sensor("left-foot-acc").data

		# cassie_right_foot_pose = self.data.sensor("right-foot-pose").data
		# cassie_right_foot_vel = self.data.sensor("right-foot-vel").data
		# cassie_right_foot_acc = self.data.sensor("right-foot-acc").data


		q_position = self.select_cassie_qpos(self.data.qpos.flat.copy())
		q_velocity = self.select_cassie_qvel(self.data.qvel.flat.copy())

		# com_inertia = self.data.cinert.flat.copy()
		# com_velocity = self.data.cvel.flat.copy()

		# actuator_forces = self.data.qfrc_actuator.flat.copy()
		# external_contact_forces = self.data.cfrc_ext.flat.copy()

		if self._exclude_current_positions_from_observation:
			q_position = q_position[2:]




		

		observations = np.concatenate((
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

									end_effector_pos,
									end_effector_ori,
									self.data_obs[6:],
									
									# cassie_joint_data,
									# cassie_pelvis_pose,
									# cassie_pelvis_vel,
									cassie_pelvis_orientation,
									cassie_pelvis_ang_vel,
									cassie_pelvis_linear_acc,
									qpos,
									qvel,
									# com_inertia,
									# com_velocity,
									# actuator_forces
									# external_contact_forces
									), dtype=np.float32)
		
		# print("observations shape",observations.shape,"\n\n\n\n\n")
		
		return {"observation": observations}

	@property
	def healthy_reward(self):
		return (
			float(self.is_healthy or self._terminate_when_unhealthy)
			* self._healthy_reward
		)

	def control_cost(self, action):
		control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
		return control_cost

	@property
	def is_healthy(self):
		min_z, max_z = self._healthy_z_range
		ee_z = self.data.sensor("end-effector-pos").data[2]
		is_healthy = min_z < self.data.qpos[2] < max_z and ee_z > 0.3

		return is_healthy

	@property
	def check_terminated(self):
		terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
		return terminated

	# def _get_obs(self):
	# 	position = self.data.qpos.flat.copy()
	# 	velocity = self.data.qvel.flat.copy()

	# 	com_inertia = self.data.cinert.flat.copy()
	# 	com_velocity = self.data.cvel.flat.copy()

	# 	actuator_forces = self.data.qfrc_actuator.flat.copy()
	# 	external_contact_forces = self.data.cfrc_ext.flat.copy()

	# 	if self._exclude_current_positions_from_observation:
	# 		position = position[2:]

	# 	return np.concatenate(
	# 		(
	# 			position,
	# 			velocity,
	# 			com_inertia,
	# 			com_velocity,
	# 			actuator_forces,
	# 			external_contact_forces,
	# 		)
	# 	)
	
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

		end_effector_pos = self.data.sensor("end-effector-pos").data
		end_effector_ori = self.data.sensor("end-effector-orientation").data


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

				end_effector_pos,
				end_effector_ori,
				self.data_obs[6:]
			)
		)
		return dict(observation= self.observations.tolist())
	
	def do_simulation(self, ctrl, n_frames):

		# TODO: implement refernce arm actions from the manipulation policy
		for _ in range(n_frames):
			self.data.ctrl[:] = ctrl
			mujoco.mj_step(self.model, self.data)

	def step(self, action):
		# xy_position_before = mass_center(self.model, self.data)
		# self.do_simulation(action, self.frame_skip)
		# xy_position_after = mass_center(self.model, self.data)

		# xy_velocity = (xy_position_after - xy_position_before) / self.dt
		# x_velocity, y_velocity = xy_velocity

		# ctrl_cost = self.control_cost(action)

		# forward_reward = self._forward_reward_weight * x_velocity
		# healthy_reward = self.healthy_reward

		# rewards = forward_reward + healthy_reward

		# observation = self._get_obs()
		# reward = rewards - ctrl_cost
		# terminated = self.terminated
		# info = {
		#     "reward_linvel": forward_reward,
		#     "reward_quadctrl": -ctrl_cost,
		#     "reward_alive": healthy_reward,
		#     "x_position": xy_position_after[0],
		#     "y_position": xy_position_after[1],
		#     "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
		#     "x_velocity": x_velocity,
		#     "y_velocity": y_velocity,
		#     "forward_reward": forward_reward,
		# }

		# if self.render_mode == "human":
		#     self.render()
		# return observation, reward, terminated, False, info

		

		# TODO: implement refernce arm actions from the manipulation policy
		""" get acitons from the policy and apply it to the mujoco model for the arm 
			action = [action_cassie, action_arm_pos + del_corrections, action_arm_vel + del_corrections]
			actions from the referfcen trajectory has torque values directed for cassie
			what i want is to actually get the reward from the difference between action torques for the initial
			cassie position
		"""

		ref_end_effector_orientation, ref_end_effector_position = self.compute_relative_pose(self.data_obs[9:], self.data_obs[6:9])

		self.arm_action_pred_online = self.get_arm_action_predicition(self.data_obs[6:9], self.data_obs[9:])
		# self.arm_action_pred = self.get_arm_action_predicition(ref_end_effector_position, ref_end_effector_orientation)

		mixing_coeff_pred = 0.01

		action_arm = self.arm_action_pred + mixing_coeff_pred * self.arm_action_pred_online
		
		# action_arm = self.get_arm_action_predicition(self.data_obs[6:9], self.data_obs[9:])

		# print("target_arm pose",self.data_obs[0:3])

		mixing_coeff = 0.1
		clipping = 0.5

		arm_bounds = np.array([3.14159, 2.40855, 2.659881, 3.14159, 2.23402, 3.14159])

		action_cassie = action[0:10]
		self.action_arms = action[10:16] # data logging
		action_arm_og = action[10:16]

		current_arm_corrections = mixing_coeff*np.clip(action_arm_og, a_min = -arm_bounds, a_max = arm_bounds)
		
		# print("\n\n\n\n\n\n\n", current_arm_corrections.shape, "\n\n\n\n\n\n\n")

		# current_arm_corrections = self.arm_corrections + mixing_coeff*np.clip(action_arm_og, a_min = -arm_bounds, a_max = arm_bounds)

		action_arm_mixed = action_arm + current_arm_corrections

		self.arm_corrections = current_arm_corrections

		actions_vel = np.zeros(6)



		action = np.concatenate((action_cassie, action_arm_mixed, actions_vel)) 

		# print("\n\n\n\n\n\n\n", action_arm_mixed.shape, "\n\n\n\n\n\n\n")

		self.do_simulation(action, self.frame_skip)

		# pelvis calculation from thje mujoco model
		pelvis_pose, pelvis_ori, pelvis_vel = self.pelvis_center()

		xy_position_after = pelvis_pose[0:2]


		# Transition happens here so time + 1
		self.timestamp += 1

		xy_velocity = pelvis_vel[0:2]
		x_velocity, y_velocity = xy_velocity

		ctrl_cost = 0.1 * self.control_cost(action)

		forward_reward = self._forward_reward_weight * x_velocity
		healthy_reward = self.healthy_reward

		rewards = forward_reward + healthy_reward

		# joint_idx = np.asarray([7, 8, 9, 14, 20, 21, 22, 23, 28, 34])
		joint_idx = np.asarray([7, 8, 9, 14, 21, 22, 23, 28])
		joint_idx_cassie = joint_idx.tolist()

		# if frameskip = 5, we don't need to multiply 6
		# reference 0 position

		# ref_qpos_cassie, ref_qvel_cassie = self.ref_trajectory.state(self.timestamp * self.frame_skip)
		# ref_qpos_cassie, ref_qvel_cassie = self.ref_trajectory.state(1)
		ref_qpos_cassie = self.init_qpos[0:35]

		# ref_qpos = self.set_qpos(ref_qpos_cassie)
		ref_qpos = ref_qpos_cassie
		# ref_qpos[2] = ref_qpos[2] + self.pelvis_z_offset

		# ref_qpos_target = ref_qpos_cassie

		ref_pelvis_pos = ref_qpos[0:3]
		ref_pelvis_ori = ref_qpos[3:7]
		ref_joint_pos = ref_qpos_cassie[joint_idx]

		current_pelvis_pos = pelvis_pose
		current_pelvis_ori = pelvis_ori
		current_joint_input = self.get_cassie_joint_pos().tolist()
		current_joint_pos = self.data.qpos[:35][joint_idx_cassie]

		current_pelvis_vel = self.data.sensor("pelvis-vel").data
		current_pelvis_acc = self.data.sensor("pelvis-linear-acceleration").data

		sensor_endeffector_position = self.data.sensor("end-effector-pos").data
		sensor_endeffector_orientation = self.data.sensor("end-effector-orientation").data
		current_endeffector_orientation, current_endeffector_position = self.compute_relative_pose(sensor_endeffector_orientation, sensor_endeffector_position)



		current_endeffector_vel = self.data.sensor("end-effector-vel").data

		# ref_end_effector_position = self.data_obs[0:3]
		# ref_end_effector_orientation = self.data_obs[3:7]
		ref_end_effector_position = self.data_obs[6:9]
		ref_end_effector_orientation = self.data_obs[9:]

		base_pos = self.data.sensor("base-link-pos").data
		base_ori = self.data.sensor("base-link-ori").data

		ref_end_effector_orientation, ref_end_effector_position = self.compute_relative_pose(self.data_obs[9:], self.data_obs[6:9])


		ref_mpos, ref_mvel, ref_torque = self.ref_trajectory.action(
			self.timestamp * self.frame_skip
		)

		cassie_action = np.asarray(action).flat[:10]

		# the following imitation reward design is from Zhaoming's 2023 paper https://zhaomingxie.github.io/projects/Opt-Mimic/opt-mimic.pdf
		# sigmas = [0.05, 0.05, 0.3, 0.35, 0.3]
		sigmas = [0.5, 1, 1, 1, 1, 1, 0.8, 0.5]
		reward_weights = [1, 0.4, 0.7, 0.1, 0.0, 2, 0.1, 0.9]

		# reward for pelvis position difference
		self.r_0 = np.exp(
			-(np.linalg.norm(ref_pelvis_pos - current_pelvis_pos, ord=2) ** 2)
			/ (2 * sigmas[0] ** 2)
		)
		
		# reward for pelvis orientation difference
		self.r_1 = np.exp(
			-(np.linalg.norm(ref_pelvis_ori - current_pelvis_ori, ord=2) ** 2)
			/ (2 * sigmas[1] ** 2)
		)
		# reward for joint position difference
		
		# self.r_2 = np.exp(
		# 	-(np.linalg.norm(ref_joint_pos - current_joint_pos, ord=2) ** 2)
		# 	/ (2 * sigmas[2] ** 2)
		# )
		
		self.r_2 = (
			-(np.linalg.norm(ref_joint_pos - current_joint_pos, ord=2) ** 2)
		)
		
		# reward for action difference
		self.r_3 = np.exp(-(np.linalg.norm(ref_torque - cassie_action, ord=2)) / (2 * sigmas[3]))
		# reward for maximum action difference
		current_max_action = np.max(np.abs(cassie_action))
		ref_max_action = np.max(np.abs(ref_torque))
		self.r_4 = np.exp(
			-(np.abs(ref_max_action - current_max_action) ** 2) / (2 * sigmas[4] ** 2)
		)
		
		# reward for end effector position difference
		self.reward_ee_pos = (
			-(np.linalg.norm(current_endeffector_position - ref_end_effector_position, ord = 2) / (3 * sigmas[5] ** 2))
		)
  
		# self.reward_ee_pos = (
		# 	-np.abs((np.linalg.norm(current_endeffector_position-ref_end_effector_position)))
		# )

		# self.reward_ee_pos = (
		# 	-np.linalg.norm((np.abs(np.square(current_endeffector_position)-np.square(ref_end_effector_position))), ord=2)
		# )

		# print("ref_ee_position", ref_end_effector_position)
		# print("current_ee_position", current_endeffector_position, "\n\n\n\n\n")
		

		# reward for end effector orientation difference
		self.reward_ee_ori = np.exp(
			-(np.linalg.norm(current_endeffector_orientation - ref_end_effector_orientation, ord = 2) / (2 * sigmas[6] ** 2))
		)

		self.del_ee_pos = np.abs((np.linalg.norm(current_endeffector_position-ref_end_effector_position)))

		self.del_ee_ori = np.linalg.norm(current_endeffector_orientation - ref_end_effector_orientation)

		self.reward_cassie_pelvis_vel = (-np.linalg.norm(current_pelvis_vel[0:2], ord=2))
		self.reward_cassie_pelvis_acc = np.exp(-np.linalg.norm(current_pelvis_acc[0:2], ord=2))

		# self.reward_cassie_qpos = np.exp(
		# 	-(np.linalg.norm(ref_qpos - self.data.qpos[:35], ord=2) ** 2) / (1* 1)
		# )

		# self.reward_cassie_qpos = np.exp(
		# 	-(np.linalg.norm(ref_qpos[joint_idx_cassie] - self.data.qpos[:35][joint_idx_cassie], ord=2) ** 2) / (1* 1)
		# )

		# r_5 = (
		#     np.exp(-(np.linalg.norm(ref_qvel - self.data.qvel, ord=2)) / (2 * 1)) * 1e1
		# )  # + np.exp(-(np.linalg.norm(ref_qpos[:-1] - self.data.qpos))) * 1e1

		target_z = ref_qpos[2]

		self.reward_z = -np.abs(target_z - current_pelvis_pos[2])
		self.reward_cassie_qpos = -np.exp(np.linalg.norm(ref_qpos[3:7] - current_pelvis_ori, ord=2) ** 2) / (1* 3)

		# self.reward_cassie_qpos = -(np.linalg.norm(ref_qpos[3:7] - pelvis_ori) ** 2) / (1* 1)
		
		self.reward_cassie_foot = self.calculate_foot_reward()

		

		# total_reward = (
		# 	reward_weights[0] * self.r_0
		# 	+ reward_weights[1] * self.r_1
		# 	+ reward_weights[2] * self.r_2
		# 	+ reward_weights[3] * self.r_3
		# 	+ reward_weights[4] * self.r_4
		# 	+ reward_weights[5] * self.reward_ee_pos
		# 	+ reward_weights[6] * self.reward_ee_ori
		# 	+ reward_weights[7] * self.reward_cassie_qpos
		# 	+ 1

		# )  # + 0.3 * r_5
		# total_reward = -np.linalg.norm(self.data.qpos - ref_qpos[:-1])-np.linalg.norm(action-ref_torque)
		
		# total_reward = np.exp(-np.linalg.norm(self.data.qpos - ref_qpos)) + np.exp(
		# 	-np.linalg.norm(action - ref_torque)
		# )

		foot_reward = self.calculate_foot_reward()

		R = (
			reward_weights[0] * self.r_0
			# + reward_weights[1] * self.r_1
			+ reward_weights[2] * self.r_2
			# reward_weights[3] * self.r_3
			# + reward_weights[4] * self.r_4
			+ reward_weights[5] * self.reward_ee_pos
			+ reward_weights[6] * self.reward_ee_ori
			# + reward_weights[7] * self.reward_cassie_qpos
			+ 0.9 * self.reward_z
			+ 1.0 * self.reward_cassie_qpos
			+ 0.9 * self.reward_cassie_pelvis_vel
			+ 0.3 * self.reward_cassie_pelvis_acc
			# + 0.1 * np.exp(-(np.linalg.norm(action[0:10], ord=2) ** 2))
			# + 0.5 * -(np.linalg.norm(self.arm_corrections,ord=2))
			+ 5
			+ foot_reward

		) 

		observation = self.get_observations()
		observation = observation["observation"]


		# change the target body pose to the data_obs

		# if not play:
		# 	self.model.body('target').pos = self.data_obs[0:3]
		# else:
		# self.model.body('target').pos = np.reshape(observation[0:3], (3,))

		self.total_reward += R
		
		reward = R

		# print("reward",reward,"\n\n\n\n\n")
		# reward = total_reward + forward_reward + healthy_reward - ctrl_cost
		# reward = forward_reward + healthy_reward - ctrl_cost
		terminated = self.check_terminated
		self.info = {
			"reward_linvel": forward_reward,
			"reward_quadctrl": -ctrl_cost,
			"reward_alive": healthy_reward,
			"x_position": xy_position_after[0],
			"y_position": xy_position_after[1],
			"distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
			"distance_from_target_ee_pose": np.linalg.norm(current_endeffector_position - ref_end_effector_position, ord=2),
			"x_velocity": x_velocity,
			"y_velocity": y_velocity,
			"forward_reward": forward_reward,
		}

		# if self.render_mode == "human":
		# 	self.render()

		# print(ref_qpos[:3], current_pelvis_pos[:3])
		# print(reward)
		# if terminated:
		# exit()
		# print(f'final x pos {xy_position_after[0]:.2f}, {ref_pelvis_pos[0]:.2f}, {current_pelvis_pos[0]:.2f}')
		# print(f'{r_0:.2e}, {r_1:.2e}, {r_2:.2e}, {r_3:.2e}, {r_4:.2e}, {r_5:.2e}')
		# print(f'{self.data.qpos[:3]}, {ref_pelvis_pos}')
		# import time
		# time.sleep(0.01)

		## render mujoco viewer	
			
		if self.render_mode == "human":
			try:
				self.mj_viewer.render()
			except:
				print("rendering failed, initialiing viewer again")
				# self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

		if self.timestamp > self.max_episode_steps:
			terminated = True

		return {"observation":observation}, reward, terminated, False, self.info
	
	def calculate_foot_reward(self, weight_in_polygon = 2, weight_foot_ori = 0.6, weight_heel_toe_z = 1.8, weight_foot_y = 0.8):


		com_pos = self.com_center_pos()
		# left foot
		left_foot_pos = self.data.sensor("left-foot-pos").data
		left_heel_pos = self.data.sensor("left-heel-pos").data
		left_toe_pos = self.data.sensor("left-toe-pos").data
		left_foot_ori = self.data.sensor("left-foot-orientation").data
		# left_foot_vel = self.data.sensor("left-foot-vel").data
		# left_foot_acc = self.data.sensor("left-foot-acc").data

		# right foot
		right_foot_pos = self.data.sensor("right-foot-pos").data
		right_heel_pos = self.data.sensor("right-heel-pos").data
		right_toe_pos = self.data.sensor("right-toe-pos").data
		right_foot_ori = self.data.sensor("right-foot-orientation").data
		# right_foot_vel = self.data.sensor("right-foot-vel").data
		# right_foot_acc = self.data.sensor("right-foot-acc").data

		# reward for com position within the support polygon

		polygon = [
			[left_heel_pos[0], left_heel_pos[1]], 
			[left_toe_pos[0], left_toe_pos[1]], 
			[right_toe_pos[0], right_toe_pos[1]], 
			[right_heel_pos[0], right_heel_pos[1]]
		]

		# area_of_polygon = PolygonArea(polygon)

		# reward for com position within the support polygon
		reward_in_polygon = 0

		if point_in_polygon(com_pos[0:2], polygon):
			reward_in_polygon += weight_in_polygon

		# reward for left foot and right foot orientation difference
			
		reward_foot_ori = (
			-(np.linalg.norm(left_foot_ori - right_foot_ori, ord=2))
		)

		left_heel_z = left_heel_pos[2]
		left_toe_z = left_toe_pos[2]
		right_heel_z = right_heel_pos[2]
		right_toe_z = right_toe_pos[2]

		# print("left_heel_z",left_heel_z,"left_toe_z",left_toe_z,"right_heel_z",right_heel_z,"right_toe_z",right_toe_z,"\n\n\n\n\n")

		# reward for heel and toe position difference in z direction
		reward_left_heel_toe_z = np.exp(
			-(np.sqrt((left_heel_z - left_toe_z)**2))
		)
		reward_right_heel_toe_z = np.exp(
			-(np.sqrt((right_heel_z - right_toe_z)**2))
		)

		# reward for left and right foot having same y position
		left_foot_y = left_foot_pos[1]
		right_foot_y = right_foot_pos[1]
		reward_foot_y = np.exp(
			-(np.sqrt((left_foot_y - right_foot_y)**2))
		)



		# reward for left foot and right foot having same orientation
		# self.reward_foot_ori = np.exp(
		# 	-(np.linalg.norm(left_foot_ori - right_foot_ori, ord=2))
		# )
  
		left_foot_vel = self.data.sensor("left-foot-vel").data
		right_foot_vel = self.data.sensor("right-foot-vel").data

		foot_reward = (
			# 0.1 * self.reward_left_foot_pos
			# + 0.1 * self.reward_left_foot_vel
			# + 0.1 * self.reward_left_foot_acc
			# + 0.1 * self.reward_right_foot_pos
			+ weight_foot_ori * reward_foot_ori
			+ weight_heel_toe_z* (reward_left_heel_toe_z + reward_right_heel_toe_z)
			+ weight_foot_y * reward_foot_y
			+ reward_in_polygon
			- 0.4 * (np.linalg.norm(left_foot_vel) + np.linalg.norm(right_foot_vel))
		)

		return foot_reward

	def reset_model(self, target_pose = None, seed = None, random = False):

		# TODO : implement the target pose for the end effector

		noise_low = -self._reset_noise_scale
		noise_high = self._reset_noise_scale

		if not random:
			self._update_init_qpos()
		else:
			r_var = np.random.choice([False,False,False,False, False])
			if r_var:
				self._update_init_qpos()
				self.get_random_data(random = False)
			else:
				self.get_random_data(random = False)
		
		self.timestamp = 0
		self.total_reward = 0


		

		

		# print("\n\n\n\n initial qpos shape", self.init_qpos.shape)
		# print("\n\n\n\n initial qvel shape", self.init_qvel.shape)
		# qpos_reset = self.set_cassie_qpos(self.init_qpos)
		# qvel_reset = self.set_qvel(self.init_qvel)

		qpos = self.data.qpos + self.np_random.uniform(
			low=noise_low, high=noise_high, size=self.model.nq
		)
		# qvel = qvel_reset + self.np_random.uniform(
		# 	low=noise_low, high=noise_high, size=self.model.nv
		# )

		qpos = self.data.qpos 
		

		# self.set_state(qpos, qvel)
		self.data.qpos[:] = np.copy(qpos)
		# self.data.qvel[:] = np.copy(qvel)
		if self.model.na == 0:
			self.data.act[:] = None

		# mujoco.mj_forward(self.model, self.data)
		
		self.timestamp = 0

		
		mujoco.mj_step(self.model, self.data)

		observation = self.get_observations()
		observation = observation["observation"]
		
		# predict arm action
		if target_pose is not None:
			self.data_obs[6:] = target_pose
		else:
			self.get_random_data(random=True)

		# print("target pose to ik",self.data_obs[6:9])
		
		base_pos = self.data.sensor("base-link-pos").data
		base_ori = self.data.sensor("base-link-ori").data
		ref_end_effector_orientation, ref_end_effector_position = self.compute_relative_pose(self.data_obs[9:], self.data_obs[6:9])

		# self.arm_action_pred = self.get_arm_action_predicition(self.data_obs[6:9], self.data_obs[9:])
		self.arm_action_pred = self.get_arm_action_predicition(ref_end_effector_position, ref_end_effector_orientation)
		
		self.arm_corrections = np.zeros(6)

		self.info = {}

			


		return {"observation": observation}, self.info
	
	def reset(self, target_pose = None, seed = None, random = False):
		target_pose = None

		# random = np.random.choice([True, False])
		if random:
			obs, info = self.reset_model(target_pose = target_pose, random = random)
		else:
			obs, info = self.reset_model(target_pose = target_pose, random = False)
		return obs, info
	
	def get_arm_action_predicition(self, ee_pos, ee_ori):
		action_tol = 1e-3
		# arm_action_pred = self.arm_policy.get_action(self.data_obs[0:3], self.data_obs[3:7])
		arm_action_pred = self.arm_policy.get_action(self.data_obs[6:9], self.data_obs[9:])
		for _ in range(10):
			# self.do_simulation(np.zeros(16,), 1)
			# arm_action_pred_n = self.arm_policy.get_action(self.data_obs[0:3], self.data_obs[3:7])
			arm_action_pred_n = self.arm_policy.get_action(self.data_obs[6:9], self.data_obs[9:])
			if np.linalg.norm(arm_action_pred - arm_action_pred_n) < action_tol:
				break

		return arm_action_pred_n

	# def viewer_setup(self):
	# 	assert self.viewer is not None
	# 	for key, value in DEFAULT_CAMERA_CONFIG.items():
	# 		if isinstance(value, np.ndarray):
	# 			getattr(self.viewer.cam, key)[:] = value
	# 		else:
	# 			setattr(self.viewer.cam, key, value)
