import isaacgym

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from satellite.envs.quadcopter import Quadcopter
from satellite.envs.wrappers.isaacgym_envs import IsaacGymWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

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

set_seed()

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}

cfg = {
    'name': 'Quadcopter',
    'physics_engine': 'physx',
    'env': {
        'numEnvs': 32768,
        'envSpacing': 1.25, 
        'maxEpisodeLength': 500, 
        'enableDebugVis': False, 
        'clipObservations': 5.0, 
        'clipActions': 1.0, 
        'enableCameraSensors': False
    },
    'sim': {
        'dt': 0.01,
        'substeps': 2,
        'up_axis': 'z', 
        'use_gpu_pipeline': True, 
        'gravity': [0.0, 0.0, -9.81], 
        'physx': {
            'num_threads': 4, 
            'solver_type': 1, 
            'use_gpu': True, 
            'num_position_iterations': 4, 
            'num_velocity_iterations': 0, 
            'contact_offset': 0.02, 
            'rest_offset': 0.001, 
            'bounce_threshold_velocity': 0.2, 
            'max_depenetration_velocity': 1000.0, 
            'default_buffer_size_multiplier': 5.0, 
            'max_gpu_contact_pairs': 1048576, 
            'num_subscenes': 4, 
            'contact_collection': 0
        }
    },
    'task': {
        'randomize': False
    }
}

env = Quadcopter(
    cfg=cfg,
    rl_device="cuda:0",
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=True,
    virtual_screen_capture=False,
    force_render=False
)

env = IsaacGymWrapper(env)

memory = RandomMemory(memory_size=8, num_envs=env.num_envs, device=env.device)

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, env.device)
models["value"] = models["policy"]  # same instance: shared model

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 8  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4  # 8 * 8192 / 16384
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
#cfg["experiment"]["write_interval"] = 20
#cfg["experiment"]["checkpoint_interval"] = 200
#cfg["experiment"]["directory"] = "runs"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)


cfg_trainer = {"timesteps": 8, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ──────────────────────────────────────────────────────────────────────────
# Setup PyTorch profiler
log_dir = "/home/andreaberti/profiler_logs/ISAAC_SKRL_Integration_base/quadcopter"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler(log_dir),
    #record_shapes=True,
    #profile_memory=True,
    #with_stack=True,
    #with_flops=True,
    #with_modules=True,
)
# ──────────────────────────────────────────────────────────────────────────

prof.start()
trainer.train()
prof.stop()

output_path = "/home/andreaberti/profiler_text/ISAAC_SKRL_Integration_base/quadcopter/text_output.txt"
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

csv_path = "/home/andreaberti/profiler_text/ISAAC_SKRL_Integration_base/quadcopter/csv_output.csv"
if not os.path.exists(os.path.dirname(csv_path)):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path, index=False)
