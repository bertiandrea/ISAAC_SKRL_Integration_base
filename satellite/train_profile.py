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

# ──────────────────────────────────────────────────────────────────────────────
# Profiler imports
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import os
import pandas as pd
# ──────────────────────────────────────────────────────────────────────────────

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

def setup_profiler(log_dir = "/home/andreaberti/profiler_logs/ISAAC_SKRL_Integration_base/satellite"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )

def save_profiler_results(prof, log_dir="/home/andreaberti/profiler_logs/ISAAC_SKRL_Integration_base/satellite"):
    output_path = log_dir + "/text_output.txt"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    events = prof.key_averages()

    with open(output_path, "w") as f:
        f.write(events.table(sort_by="self_cuda_time_total", row_limit=500))

        f.write("\n\n\n")

        f.write(events.table(sort_by="self_cpu_time_total", row_limit=500))

        f.write("\n\n\n")

        f.write(events.table(sort_by="self_cuda_memory_usage", row_limit=500))

        f.write("\n\n\n")

        f.write(events.table(sort_by="self_cpu_memory_usage", row_limit=500))

    rows = []
    for e in events:
        rows.append({
            "name":               e.key[:50],  # Truncate to 50 characters
            "self_cpu_time_ms":   e.self_cpu_time_total / 1e3,
            "cpu_time_ms":        e.cpu_time_total / 1e3,

            "self_cuda_time_ms":  e.self_device_time_total / 1e3,
            "cuda_time_ms":       e.device_time_total / 1e3,

            "self_cpu_memory_bytes":   e.self_cpu_memory_usage,
            "self_cuda_memory_bytes":  e.self_device_memory_usage,

            "cpu_memory_bytes":   e.cpu_memory_usage,
            "cuda_memory_bytes":  e.device_memory_usage,

            "count":              e.count,
            "flops":              e.flops,

            "device_type":        str(e.device_type),
        })
    df = pd.DataFrame(rows)
    
    df['order'] = df['name'].str[0].map({'#': 0, '$': 1}).fillna(2).astype(int)
    df = df.sort_values(['order', 'name'], ascending=[True, True])
    df = df.drop(columns='order')

    print(df.head(40))

    csv_path = log_dir + "/csv_output.csv"
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

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
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
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

    prof = setup_profiler()

    prof.start()
    trainer.train()
    prof.stop()

    save_profiler_results(prof)


if __name__ == "__main__":
    main()