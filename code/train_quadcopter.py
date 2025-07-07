import numpy as np
import isaacgym

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from code.envs.quadcopter import Quadcopter
from code.envs.wrappers.isaacgym_envs import IsaacGymWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

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

#####################################################################################

cfg = {
    'physics_engine': 'physx',
    'env': {
        'numEnvs': 4096,
        'envSpacing': 1.25,        
        'clipObservations': 5.0,  # no clipping
        'clipActions': 1.0,  # no clipping
        'enableCameraSensors': False
    },
    'sim': {
        'dt': 0.0166,
        'substeps': 2,
        'up_axis': 'z', 
        'use_gpu_pipeline': True, 
        'gravity': [0.0, 0.0, -9.81], 
        'physx': {
            'use_gpu': True, 
        },
        'num_position_iterations': 4,
        'num_velocity_iterations': 0,
        'contact_offset': 0.02,
        'rest_offset': 0.001,
        'bounce_threshold_velocity': 0.2,
        'max_depenetration_velocity': 1000.0,
        'default_buffer_size_multiplier': 5.0,
        'max_gpu_contact_pairs': 1048576, # 1024*1024
        'contact_collection': 0
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
    headless=False,
    virtual_screen_capture=False,
    force_render=False
)

env = IsaacGymWrapper(env)

#####################################################################################

memory = RandomMemory(memory_size=8, num_envs=env.num_envs, device=env.device)

models = {}
models["policy"] = Shared(env.observation_space, env.action_space, env.device)
models["value"] = models["policy"] 

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 8  # memory_size
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 4  # 8 * 8192 / 16384
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 1e-3
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
cfg_ppo["experiment"]["write_interval"] = 'auto'
cfg_ppo["experiment"]["checkpoint_interval"] = 'auto'
cfg_ppo["experiment"]["directory"] = "runs/quadcopter"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)


cfg_trainer = {"timesteps": 1000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()