import numpy as np
import torch

from satellite.utils.satellite_util import sample_random_quaternion_batch, quat_diff, quat_diff_rad
from satellite.envs.vec_task import VecTask
from satellite.rewards.satellite_reward import (
    TestReward,
    TestRewardSmooth,
    RewardFunction
)
from satellite.pid.pid import PID
from satellite.controller.controller import SatelliteAttitudeController

from isaacgym import gymutil, gymtorch, gymapi

from torch.profiler import record_function

class Satellite(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, reward_fn: RewardFunction = None):
        self.cfg = cfg

        self.dt = cfg["sim"].get('dt', 1 / 60.0)  # seconds
        self.env_spacing = cfg["env"].get('envSpacing', 0.0)
        self.asset_name = cfg["env"]["asset"].get('assetName', 'satellite')
        self.asset_root = cfg["env"]["asset"].get('assetRoot', '/home/andreaberti/ISAAC_SKRL_Integration_base/satellite') 
        self.asset_file = cfg["env"]["asset"].get('assetFileName', 'satellite.urdf')
        self.asset_init_pos_p = cfg["env"]["asset"].get('init_pos_p', [0.0, 0.0, 0.0])
        self.asset_init_pos_r = cfg["env"]["asset"].get('init_pos_r', [0.0, 0.0, 0.0, 1.0])
        self.actuation_noise_std = cfg["env"].get('actuation_noise_std', 0.0)
        self.sensor_noise_std = cfg["env"].get('sensor_noise_std', 0.0)
        self.torque_scale = cfg["env"].get('torque_scale', 1.0)
        self.threshold_ang_goal = cfg["env"].get('threshold_ang_goal', 0.01745)  # radians
        self.threshold_vel_goal = cfg["env"].get('threshold_vel_goal', 0.01745)  # radians/sec
        self.overspeed_ang_vel = cfg["env"].get('overspeed_ang_vel', 0.78540)  # radians/sec
        self.max_episode_length = cfg["env"].get('episode_length_s', 120) / self.dt  # seconds
        
        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        ################# SETUP SIM #################
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self.num_envs, 13)
        self.satellite_pos     = self.root_states[:, 0:3]
        self.satellite_quats   = self.root_states[:, 3:7]
        self.satellite_linvels = self.root_states[:, 7:10]
        self.satellite_angvels = self.root_states[:, 10:13]
        #############################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = self.root_states.clone()
        self.prev_angvel = self.satellite_angvels.clone()
        ########################################

        self.goal_quat = sample_random_quaternion_batch(self.device, self.num_envs)
        self.goal_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_ang_acc = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.torque_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.root_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int) * self.num_bodies
        self.force_tensor = torch.zeros_like(self.torque_tensor, device=self.device)

        if reward_fn is None:
            self.reward_fn: RewardFunction = TestReward()
        else:
            self.reward_fn = reward_fn

        self.controller_logic = cfg["controller"].get("controller_logic", False)
        if self.controller_logic:
            self.pid_rate = PID(
                num_envs=self.num_envs,
                kp=cfg["pid"]["rate"].get("kp", 1.0),
                ki=cfg["pid"]["rate"].get("ki", 0.1),
                kd=cfg["pid"]["rate"].get("kd", 0.01),
                dt=self.dt,
                device=self.device
            )
            self.controller = SatelliteAttitudeController(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.dt,
                pid_rate=self.pid_rate,
                torque_tau=cfg["controller"].get("torque_tau", 0.02)
            )

    def create_sim(self) -> None:
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(self.device_id, self.device_id, self.physics_engine, self.sim_params)
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.create_actor(i, env, self.asset, self.asset_init_pos_p, self.asset_init_pos_r, 1, self.asset_name)
    
    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)

    ################################################################################################################################
           
    def reset_idx(self, ids: torch.Tensor) -> None:
        with record_function("$SatelliteVec__reset_idx__sim"):
            ################# SIM #################
            self.root_states[ids] = self.initial_root_states[ids]
            idx32 = ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, self.actor_root_state, gymtorch.unwrap_tensor(idx32), len(idx32)
            )
            #######################################

            ################# SIM #################
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.prev_angvel = self.satellite_angvels.clone()
            ########################################
                
        with record_function("$SatelliteVec__reset_idx__sample_goal"):
            self.goal_quat[ids] = sample_random_quaternion_batch(self.device, len(ids))

        with record_function("$SatelliteVec__reset_idx__reset_buffers"):
            self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
            self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

            self.progress_buf[ids] = 0
            self.reset_buf[ids] = False
            self.timeout_buf[ids] = False

            self.rew_buf[ids] = 0.0
        
        #if self.controller_logic:
        #    self.controller.reset(ids)

    ################################################################################################################################

    def apply_torque(self, actions: torch.Tensor) -> None:
        ############## CONTROLLER ###############
        #if self.controller_logic:
        #    actions = self.controller.compute_control(
        #        actions=actions, 
        #        measured_angvels=self.satellite_angvels,
        #    )
        #########################################

        with record_function("$SatelliteVec__apply_torque__noise_and_clamp"):
            if self.actuation_noise_std > 0.0:
                actions = torch.add(
                    actions,
                    torch.normal(mean=0.0, std=self.actuation_noise_std, size=actions.shape, device=self.device)
                )
            
            self.actions = torch.mul(
                torch.clamp(actions, -self.clip_actions, self.clip_actions),
                self.torque_scale
            )

        ################# SIM #################
        with record_function("$SatelliteVec__apply_torque__sim"):
            self.torque_tensor[self.root_indices] = self.actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.force_tensor),  
                gymtorch.unwrap_tensor(self.torque_tensor), 
                gymapi.ENV_SPACE
            )
        #######################################
    
    def termination(self) -> None:
        with record_function("$SatelliteVec__termination"):      
            ids  = torch.nonzero(torch.logical_or(self.reset_buf, self.timeout_buf), as_tuple=False).flatten()
            if len(ids) > 0:
                self.reset_idx(ids)
                
    def compute_observations(self) -> None:
        ################# SIM #################
        with record_function("$SatelliteVec__compute_observations__sim"):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.satellite_angacc = torch.div(
                torch.sub(self.satellite_angvels, self.prev_angvel),
                self.dt
            )
        with record_function("$SatelliteVec__compute_observations__compute_buffers"):
            self.prev_angvel = self.satellite_angvels.clone()
            self.obs_buf = torch.cat(
                (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), self.satellite_angacc, self.actions), dim=-1)
            self.states_buf = torch.cat(
                (self.obs_buf, self.satellite_angvels), dim=-1)
        ########################################

        with record_function("$SatelliteVec__compute_observations__noise_and_clamp"):
            if self.sensor_noise_std > 0.0:
                noise = torch.normal(mean=0.0, std=self.sensor_noise_std, size=self.state_space.shape, device=self.device)
                self.obs_buf = torch.add(self.obs_buf, noise[:, :self.num_observations])
                self.states_buf = torch.add(self.states_buf, noise[:, :self.num_states])
            
            self.obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)
            self.states_buf = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs)

    def compute_reward(self) -> None:
        with record_function("$SatelliteVec__compute_reward"):
            self.rew_buf = self.reward_fn.compute(
                self.satellite_quats, self.satellite_angvels, self.satellite_angacc,
                self.goal_quat, self.goal_ang_vel, self.goal_ang_acc,
                self.actions
            )

    def check_termination(self) -> None:
        with record_function("$SatelliteVec__check_termination"):
            angle_diff = quat_diff_rad(self.satellite_quats, self.goal_quat)
            ang_vel_diff = torch.norm(
                torch.sub(self.satellite_angvels, self.goal_ang_vel),
                dim=1
            )
            goal = torch.logical_and(
                torch.lt(angle_diff, self.threshold_ang_goal),
                torch.lt(ang_vel_diff, self.threshold_vel_goal)
            )
            
            timeout = torch.ge(self.progress_buf, self.max_episode_length)

            overspeed = torch.ge(
                torch.norm(self.satellite_angvels, dim=1),
                self.overspeed_ang_vel
            )

            self.timeout_buf = torch.where(torch.logical_or(timeout, overspeed), True, False)
            self.reset_buf = torch.where(goal, True, False)

    def pre_physics_step(self, actions):
        self.apply_torque(actions)

    def post_physics_step(self):
        self.progress_buf += 1

        self.termination()
        self.compute_observations()
        self.compute_reward()
        self.check_termination()