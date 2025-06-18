import isaacgym
import torch
import argparse

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from satellite.configs.satellite_config import SatelliteConfig
from satellite.envs.satellite import Satellite
from satellite.models.custom_model import Policy, Value
from satellite.envs.wrappers.isaacgym_envs import IsaacGymWrapper
from satellite.rewards.satellite_reward import (
    TestReward,
    TestRewardSmooth,
    WeightedSumReward,
    TwoPhaseReward,
    ExponentialStabilizationReward,
    ContinuousDiscreteEffortReward,
    ShapingReward,
)

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

REWARD_MAP = {
    "test": TestReward,
    "test_smooth": TestRewardSmooth,
    "weighted_sum": WeightedSumReward,
    "two_phase": TwoPhaseReward,
    "exp_stabilization": ExponentialStabilizationReward,
    "continuous_discrete_effort": ContinuousDiscreteEffortReward,
    "shaping": ShapingReward,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training con reward function selezionabile")
    parser.add_argument(
        "--reward-fn",
        choices=list(REWARD_MAP.keys()),
        default="test",
        help="Which RewardFunction?"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    cfg_class = SatelliteConfig()
    cfg = class_to_dict(cfg_class)

    if cfg["set_seed"]:
        set_seed(cfg["seed"])
   
    env = Satellite(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        virtual_screen_capture=False,
        force_render=True,
        reward_fn=REWARD_MAP[args.reward_fn]()
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=cfg["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, env.device)
    models["value"] = Value(env.observation_space, env.action_space, env.device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    cfg["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    cfg["rl"]["PPO"]["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["rl"]["PPO"]["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
    cfg["rl"]["PPO"]["state_preprocessor"] = RunningStandardScaler
    cfg["rl"]["PPO"]["value_preprocessor"] = RunningStandardScaler
    cfg["rl"]["PPO"]["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
    
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)


    trainer = SequentialTrainer(cfg=cfg["rl"]["trainer"], env=env, agents=agent)

    trainer.train()
    
if __name__ == "__main__":
    main()