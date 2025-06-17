# satellite_reward.py

from satellite.utils.satellite_util import quat_diff_rad

import isaacgym #BugFix
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

from abc import ABC, abstractmethod
import math

class RewardFunction(ABC):
    @abstractmethod
    def compute(self,
                quats: torch.Tensor,
                ang_vels: torch.Tensor,
                ang_accs: torch.Tensor,
                goal_quat: torch.Tensor,
                goal_ang_vel: torch.Tensor,
                goal_ang_acc: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Compute reward given state and actions.
        Must be implemented by subclasses.
        """
        pass

class TestReward(RewardFunction):
    def __init__(self, alpha_q=1.0, alpha_omega=0.5, alpha_acc=0.2):
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        acc_err = torch.norm(ang_accs - goal_ang_acc, dim=1)

        r_q = self.alpha_q * (1.0 / (1.0 + phi))
        weight = 1.0 / (1.0 + phi)
        r_omega = self.alpha_omega * weight * (1.0 / (1.0 + omega_err))
        r_acc = self.alpha_acc * weight * (1.0 / (1.0 + acc_err))

        return r_q + r_omega + r_acc

class WeightedSumReward(RewardFunction):
    def __init__(self,
                 alpha_q=1.0, alpha_omega=0.3, alpha_acc=0.1,
                 q_threshE=1e-2, omega_threshE=1e-2,
                 q_threshL=1e-2, omega_threshL=1e-2,
                 bonus_q=200.0, bonus_stable=1000.0,
                 penalty_lvl1=-10.0, penalty_lvl2=-50.0,
                 action_saturation_thresh=None, penalty_saturation=-10.0):
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc
        self.q_threshE = q_threshE
        self.omega_threshE = omega_threshE
        self.q_threshL = q_threshL
        self.omega_threshL = omega_threshL
        self.bonus_q = bonus_q
        self.bonus_stable = bonus_stable
        self.penalty_lvl1 = penalty_lvl1
        self.penalty_lvl2 = penalty_lvl2
        self.action_saturation_thresh = action_saturation_thresh
        self.penalty_saturation = penalty_saturation

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        acc_err = torch.norm(ang_accs - goal_ang_acc, dim=1)

        base = (
            self.alpha_q * (1.0 / (1.0 + phi)) +
            self.alpha_omega * (1.0 / (1.0 + omega_err)) +
            self.alpha_acc * (1.0 / (1.0 + acc_err))
        )
        mask1 = (phi <= self.q_threshE).float()
        mask2 = ((phi <= self.q_threshE) & (omega_err <= self.omega_threshE)).float()
        mask3 = (((phi >= self.q_threshL) | (omega_err >= self.omega_threshL))).float()
        mask4 = (((phi >= 2.0 * self.q_threshL) | (omega_err >= 2.0 * self.omega_threshL))).float()
        bonus = (
            mask1 * self.bonus_q +
            mask2 * self.bonus_stable +
            mask3 * self.penalty_lvl1 +
            mask4 * self.penalty_lvl2
        )
        if self.action_saturation_thresh is not None:
            sat_mask = torch.any(actions.abs() >= self.action_saturation_thresh, dim=1).float()
            bonus = bonus + sat_mask * self.penalty_saturation

        return base + bonus

class TwoPhaseReward(RewardFunction):
    def __init__(self,
                 threshold=math.radians(1.0),
                 r1_pos=0.1, r1_neg=-0.1,
                 alpha=1.0, beta=0.5):
        self.threshold = threshold
        self.r1_pos = r1_pos
        self.r1_neg = r1_neg
        self.alpha = alpha
        self.beta = beta
        self.prev_phi = None

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)

        if self.prev_phi is None:
            r1 = torch.zeros_like(phi)
        else:
            delta = phi - self.prev_phi
            r1 = torch.where(delta < 0.0, self.r1_pos, self.r1_neg)
        r2 = self.alpha * torch.exp(-phi / self.beta)

        self.prev_phi = phi.clone()

        return torch.where(phi >= self.threshold, r2, r1)

class ExponentialStabilizationReward(RewardFunction):
    def __init__(self,
                 scale=0.14 * 2.0 * math.pi,
                 bonus=9.0,
                 goal_deg=0.25):
        self.scale = scale
        self.bonus = bonus
        self.goal_rad = math.radians(goal_deg)
        self.prev_phi = None

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        
        exp_term = torch.exp(-phi / self.scale)
        if self.prev_phi is None:
            r = exp_term
        else:
            delta = phi - self.prev_phi
            r = torch.where(delta > 0.0, exp_term, exp_term - 1.0)
        bonus = (phi <= self.goal_rad).float() * self.bonus

        self.prev_phi = phi.clone()

        return r + bonus

class ContinuousDiscreteEffortReward(RewardFunction):
    def __init__(
        self,
        error_thresh=1e-2,
        bonus=5.0,
        effort_penalty=0.1,
        fail_thresh=4.0,
        fail_penalty=-100.0
    ):
        self.error_thresh = error_thresh
        self.bonus = bonus
        self.effort_penalty = effort_penalty
        self.fail_thresh = fail_thresh
        self.fail_penalty = fail_penalty

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        
        u_sq = torch.sum(actions.pow(2), dim=1)
        sup_err = torch.max(phi, omega_err)
        r1 = -(phi + omega_err + self.effort_penalty * u_sq)
        r2 = (sup_err <= self.error_thresh).float() * self.bonus
        r3 = (sup_err >= self.fail_thresh).float() * self.fail_penalty

        return r1 + r2 + r3

class ShapingReward(RewardFunction):
    def __init__(self, mode='R4'):
        assert mode in ['R1', 'R2', 'R3', 'R4'], "Unsupported mode"
        self.mode = mode
        if mode in ['R1', 'R2']:
            self.beta_fn = lambda d: torch.where(d > 0.0, 0.5, 1.0)
        else:
            self.beta_fn = lambda d: torch.exp(-0.5 * (math.pi + d))
        if mode in ['R1', 'R3']:
            self.tau_fn = lambda phi: torch.exp(2.0 - phi.abs())
        else:
            self.tau_fn = lambda phi: 14.0 / (1.0 + torch.exp(2.0 * phi.abs()))
        self.prev_phi = None

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)

        if self.prev_phi is None:
            delta = torch.zeros_like(phi)
        else:
            delta = phi - self.prev_phi
        beta = self.beta_fn(delta)
        tau = self.tau_fn(phi)

        self.prev_phi = phi.clone()

        return beta * tau
