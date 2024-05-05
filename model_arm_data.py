import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import mujoco
import mujoco_viewer
import json

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


class ArmEnv():
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

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(669,), dtype=np.float64
        )
        self.frame_skip = 60

        self.xml_string = open(xml_string, 'r').read()

        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)

        self.timestamp = 0

        self.ee_vel_lim = 1e-4

        self.datafile = datafile

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
    
    def write_observations_to_json(self,actions, obs, filename):
        # Create a dictionary to hold the data
        data = {
            "actions": actions,
            "end-effector-position": obs[12:15].tolist(),
            "end-effector-orientation": obs[15:].tolist(),
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

        end_effector_pos = self.data.sensor("end-effector-pos").data
        end_effector_ori = self.data.sensor("end-effector-orientation").data


        return np.concatenate(
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
                end_effector_ori

            )
        )
    
    def do_simulation(self, ctrl, n_frames):
        for _ in range(n_frames):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def step(self, action, write_to_file = False):


        done = False

        

        self.do_simulation(action, self.frame_skip)

        ee_vel = self.data.sensor("end-effector-vel").data

        ee_vel_norm = np.sqrt(np.sum(ee_vel**2))

        # print(f'ee_vel_norm: {ee_vel_norm:.6e}')

        action = action[0:6]

        observation = self._get_obs()

        if ee_vel_norm < self.ee_vel_lim:
            # print("\n\nwriting to file\n\n")
            # print(" action: ",action.copy())
            if write_to_file:
                self.write_observations_to_json(action, observation, self.datafile)
            # self.write_observations_to_json(action, observation, self.datafile)
            # self.reset_model()
            done = True

        else:
            done = False
        
        # if self.render_mode == "human":
        #     self.render()
            
        if self.render_mode == "human":
            try:
                self.viewer.render()
            except:
                print("rendering failed, initialiing viewer again")
                # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # print(ref_qpos[:3], current_pelvis_pos[:3])
        # print(reward)
        # if terminated:
        # exit()
        # print(f'final x pos {xy_position_after[0]:.2f}, {ref_pelvis_pos[0]:.2f}, {current_pelvis_pos[0]:.2f}')
        # print(f'{r_0:.2e}, {r_1:.2e}, {r_2:.2e}, {r_3:.2e}, {r_4:.2e}, {r_5:.2e}')
        # print(f'{self.data.qpos[:3]}, {ref_pelvis_pos}')
        # import time
        # time.sleep(0.01)
                
        self.timestamp += 1

        if self.timestamp >= 500:
            done = True

        return observation, None, None, done, None

    def reset_model(self):
        self._update_init_qpos()
        observation = self._get_obs()
        self.timestamp = 0
        return observation
    
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
        ee_pos = obs[12:15]
        ee_ori = obs[15:]
        return ee_pos, ee_ori

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
