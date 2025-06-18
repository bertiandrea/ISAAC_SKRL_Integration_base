import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from satellite.envs.cartpole import Cartpole
from satellite.models.custom_model import Policy, Value
from satellite.envs.wrappers.isaacgym_envs import IsaacGymPreview3Wrapper

set_seed()

cfg = {
    'name': 'Cartpole',
    'physics_engine': 'physx',
    'env': {
        'numEnvs': 256,
        'envSpacing': 4.0,
        'resetDist': 3.0,
        'maxEffort': 400.0,
        'clipObservations': 5.0,
        'clipActions': 1.0,
        'asset': {
            'assetRoot': '../',
            'assetFileName': 'cartpole.urdf'
        }, 
        'enableCameraSensors': False
    },
    'sim': {
        'dt': 0.0166,
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
            'max_depenetration_velocity': 100.0,
            'default_buffer_size_multiplier': 2.0,
            'max_gpu_contact_pairs': 1048576,
            'num_subscenes': 4,
            'contact_collection': 0
        }
    },
    'task': {
        'randomize': False
    }
}

env = Cartpole(
    cfg=cfg,
    rl_device="cuda:0",
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=True,
    virtual_screen_capture=False,
    force_render=False
)
    
env = IsaacGymPreview3Wrapper(env)

memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=env.device)

models = {}
models["policy"] = Policy(env.observation_space, env.action_space, env.device)
models["value"] = Value(env.state_space, env.action_space, env.device)


cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 1  # 16 * 512 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
cfg["experiment"]["write_interval"] = 16
cfg["experiment"]["checkpoint_interval"] = 80
cfg["experiment"]["directory"] = "runs"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)


cfg_trainer = {"timesteps": 1600, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()


