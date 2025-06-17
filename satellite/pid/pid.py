# pid.py

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

class PID():
    def __init__(self, num_envs: int, device: torch.device, dt: float, Kp: float, Ki: float, Kd: float,
                 alpha: float = 0.9, clamp_d: float = 10.0, clamp_i: float = 15.0, clamp_u: float = 50.0) -> None:
        self.dt = dt
        self.device = device

        self.Kp = torch.full((num_envs, 3), Kp, device=device)
        self.Ki = torch.full((num_envs, 3), Ki, device=device)
        self.Kd = torch.full((num_envs, 3), Kd, device=device)

        self.integral = torch.zeros((num_envs, 3), dtype=torch.float, device=device)
        self.prev_feedback = torch.zeros_like(self.integral)
        self.prev_lpf_feedback = torch.zeros_like(self.integral)

        self.alpha = alpha
        self.clamp_d = clamp_d
        self.clamp_i = clamp_i
        self.clamp_u = clamp_u
    
    def update(self, error: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        # Proportional action
        p_term = torch.matmul(self.Kp, error)
        
        # Derivative action (with low-pass filter)
        lpf_fb = self.alpha*feedback + (1-self.alpha)*self.prev_feedback
        d_term = torch.matmul(self.Kd, (lpf_fb - self.prev_lpf_feedback) / self.dt) 
        d_term = torch.clamp(d_term, -self.clamp_d, self.clamp_d)

        # Integral action
        self.integral += error * self.dt
        i_term = torch.matmul(self.Ki, self.integral)
        i_term = torch.clamp(i_term, -self.clamp_i, self.clamp_i)

        u = p_term + i_term + d_term
        u = torch.clamp(u, -self.clamp_u, self.clamp_u)
   
        self.prev_lpf_feedback[:] = lpf_fb
        self.prev_feedback[:] = feedback

        return u
    
    def reset(self, env_ids):
        self.prev_feedback[env_ids] = 0.0
        self.prev_lpf_feedback[env_ids] = 0.0
        self.integral[env_ids] = 0.0