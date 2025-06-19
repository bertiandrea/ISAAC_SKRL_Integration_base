# controller.py

import isaacgym #BugFix
import torch

class SatelliteAttitudeController:
    def __init__(self, num_envs, device, dt, pid_rate, torque_tau):
        self.device = device
        self.num_envs = num_envs
        self.dt = dt
        self.pid_rate = pid_rate
        self.torque_tau = torque_tau
        self.prev_torque = torch.zeros((num_envs, 3), device=device, dtype=torch.float)

    def compute_control(self, actions: torch.Tensor, measured_angvels: torch.Tensor) -> torch.Tensor:
        rate_error = actions - measured_angvels
        raw_torque = self.pid_rate.update(rate_error, measured_angvels)

        # Apply low-pass filter to the torque command
        torque_cmd = self.torque_tau * raw_torque + (1 - self.torque_tau) * self.prev_torque
        
        self.prev_torque = torque_cmd

        return torque_cmd

    def reset(self, env_ids):
        self.prev_torque[env_ids] = 0.0
        self.pid_rate.reset(env_ids)
