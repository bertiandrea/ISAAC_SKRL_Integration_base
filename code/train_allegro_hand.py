import numpy as np
import isaacgym

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from code.envs.allegro_hand_dextreme import AllegroHandDextremeADR
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

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
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
        'numEnvs': 0.75,
        'envSpacing': 1.25,        
        'clipObservations': 5.0,  # no clipping
        'clipActions': 1.0,  # no clipping
        'enableCameraSensors': False,

        'episodeLength': 320,
        'resetTime': 8, # Max time till reset, in seconds, if a goal wasn't achieved. Will overwrite the episodeLength if is > 0.
        'enableDebugVis': False,
        'aggregateMode': 1,
        'discreteActions': False,
        'stiffnessScale': 1.0,
        'forceLimitScale': 1.0,
        'useRelativeControl': False,
        'dofSpeedScale': 20.0,
        'use_capped_dof_control': False,
        'max_dof_radians_per_second': 6.2832,
        'max_effort': 0.5,
        'num_success_hold_steps': 0,
        'actionsMovingAverage': {
            'range': [0.15, 0.2],
            'schedule_steps': 1000_000,
            'schedule_freq': 500
        },
        'controlFrequencyInv': 2,
        'cubeObsDelayProb': 0.3,
        'maxObjectSkipObs': 2,
        'actionDelayProbMax': 0.3,
        'actionLatencyMax': 15,
        'actionLatencyScheduledSteps': 2_000_000,
        'startPositionNoise': 0.01,
        'startRotationNoise': 0.0,
        'resetPositionNoise': 0.03,
        'resetPositionNoiseZ': 0.01,
        'resetRotationNoise': 0.0,
        'resetDofPosRandomInterval': 0.2,
        'resetDofVelRandomInterval': 0.0,
        'startObjectPoseDY': -0.15,
        'startObjectPoseDZ': 0.06,
        'forceScale': 2.0,
        'forceProbRange': [0.001, 0.1],
        'forceDecay': 0.99,
        'forceDecayInterval': 0.08,
        'random_network_adversary': {
            'enable': True,
            'weight_sample_freq': 1000
        },
        'random_cube_observation': {
            'enable': True,
            'prob': 0.3
        },
        'distRewardScale': -10.0,
        'rotRewardScale': 1.0,
        'rotEps': 0.1,
        'actionPenaltyScale': -0.001,
        'actionDeltaPenaltyScale': -0.2,
        'reachGoalBonus': 250,
        'fallDistance': 0.24,
        'fallPenalty': 0.0,
        'objectType': "block",
        'observationType': "no_vel",
        'asymmetric_observations': True,
        'successTolerance': 0.1,
        'printNumSuccesses': False,
        'maxConsecutiveSuccesses': 50,
        'asset': {
            'assetFileName': "urdf/kuka_allegro_description/allegro_touch_sensor.urdf",
            'assetFileNameBlock': "urdf/objects/cube_multicolor_allegro.urdf",
            'assetFileNameEgg': "mjcf/open_ai_assets/hand/egg.xml",
            'assetFileNamePen': "mjcf/open_ai_assets/hand/pen.xml"
        }
    },
    'task': {
        'randomize': True,
        'randomization_params': {
            'frequency': 720,
            'sim_params': {
                'gravity': {
                    'range': [0, 0.6],
                    'operation': "additive",
                    'distribution': "gaussian"
                }
            },
            'actor_params': {
                'hand': {
                    'scale': {
                        'range': [0.95, 1.05],
                        'operation': "scaling",
                        'distribution': "uniform",
                        'setup_only': True,
                        'color': True
                    },
                    'dof_properties':{
                        'damping': {
                            'range': [0.01, 20.0],
                            'operation': "scaling",
                            'distribution': "loguniform"
                        },
                        'stiffness': {
                            'range': [0.01, 20.0],
                            'operation': "scaling",
                            'distribution': "loguniform"
                        },
                        'effort': {
                            'range': [0.4, 10.0],
                            'operation': "scaling",
                            'distribution': "uniform"
                        },
                        'friction': {
                            'range': [0.0, 10.0],
                            'operation': "scaling",
                            'distribution': "uniform"
                        },
                        'armature': {
                            'range': [0.0, 10.0],
                            'operation': "scaling",
                            'distribution': "uniform"
                        },
                        'lower': {
                            'range': [-5.0, 5.0],
                            'operation': "additive",
                            'distribution': "uniform"
                        },
                        'upper': {
                            'range': [-5.0, 5.0],
                            'operation': "additive",
                            'distribution': "uniform"
                        }
                    },
                    'rigid_body_properties': {
                        'mass': {
                            'range': [0.4, 1.6],
                            'operation': "scaling",
                            'distribution': "uniform",
                            'setup_only': False
                        }
                    },
                    'rigid_shape_properties': {
                        'friction': {
                            'num_buckets': 250,
                            'range': [0.01, 2.0],
                            'operation': "scaling",
                            'distribution': "uniform"
                        },
                        'restitution': {
                            'num_buckets': 100,
                            'range': [0.0, 0.5],
                            'operation': "additive",
                            'distribution': "uniform"
                        }
                    }
                },
                'object': {
                    'scale': {
                        'range': [0.95, 1.05],
                        'operation': "scaling",
                        'distribution': "uniform",
                        'setup_only': True
                    },
                    'rigid_body_properties': {
                        'mass': {
                            'range': [0.3, 1.7],
                            'operation': "scaling",
                            'distribution': "uniform",
                            'setup_only': False
                        }
                    },
                    'rigid_shape_properties': {
                        'friction': {
                            'num_buckets': 250,
                            'range': [0.01, 2.0],
                            'operation': "scaling",
                            'distribution': "uniform"
                        },
                        'restitution': {
                            'num_buckets': 100,
                            'range': [0.0, 0.5],
                            'operation': "additive",
                            'distribution': "uniform"
                        }
                    }
                }
            }
        },
        "adr": {
            "use_adr": True,
            "update_adr_ranges": True,
            "clear_other_queues": False,
            "adr_extended_boundary_sample": False,
            "worker_adr_boundary_fraction": 0.4,
            "adr_queue_threshold_length": 256,
            "adr_objective_threshold_low": 5,
            "adr_objective_threshold_high": 20,
            "adr_rollout_perf_alpha": 0.99,
            "adr_load_from_checkpoint": False,
            "params": {
                "hand_damping": {
                    "range_path": "actor_params.hand.dof_properties.damping.range",
                    "init_range": [0.5, 2.0],
                    "limits": [0.01, 20.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_stiffness": {
                    "range_path": "actor_params.hand.dof_properties.stiffness.range",
                    "init_range": [0.8, 1.2],
                    "limits": [0.01, 20.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_joint_friction": {
                    "range_path": "actor_params.hand.dof_properties.friction.range",
                    "init_range": [0.8, 1.2],
                    "limits": [0.0, 10.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_armature": {
                    "range_path": "actor_params.hand.dof_properties.armature.range",
                    "init_range": [0.8, 1.2],
                    "limits": [0.0, 10.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_effort": {
                    "range_path": "actor_params.hand.dof_properties.effort.range",
                    "init_range": [0.9, 1.1],
                    "limits": [0.4, 10.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_lower": {
                    "range_path": "actor_params.hand.dof_properties.lower.range",
                    "init_range": [0.0, 0.0],
                    "limits": [-5.0, 5.0],
                    "delta": 0.02,
                    "delta_style": "additive"
                },
                "hand_upper": {
                    "range_path": "actor_params.hand.dof_properties.upper.range",
                    "init_range": [0.0, 0.0],
                    "limits": [-5.0, 5.0],
                    "delta": 0.02,
                    "delta_style": "additive"
                },
                "hand_mass": {
                    "range_path": "actor_params.hand.rigid_body_properties.mass.range",
                    "init_range": [0.8, 1.2],
                    "limits": [0.01, 10.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_friction_fingertips": {
                    "range_path": "actor_params.hand.rigid_shape_properties.friction.range",
                    "init_range": [0.9, 1.1],
                    "limits": [0.1, 2.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "hand_restitution": {
                    "range_path": "actor_params.hand.rigid_shape_properties.restitution.range",
                    "init_range": [0.0, 0.1],
                    "limits": [0.0, 1.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "object_mass": {
                    "range_path": "actor_params.object.rigid_body_properties.mass.range",
                    "init_range": [0.8, 1.2],
                    "limits": [0.01, 10.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "object_friction": {
                    "range_path": "actor_params.object.rigid_shape_properties.friction.range",
                    "init_range": [0.4, 0.8],
                    "limits": [0.01, 2.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "object_restitution": {
                    "range_path": "actor_params.object.rigid_shape_properties.restitution.range",
                    "init_range": [0.0, 0.1],
                    "limits": [0.0, 1.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "cube_obs_delay_prob": {
                    "init_range": [0.0, 0.05],
                    "limits": [0.0, 0.7],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "cube_pose_refresh_rate": {
                    "init_range": [1.0, 1.0],
                    "limits": [1.0, 6.0],
                    "delta": 0.2,
                    "delta_style": "additive"
                },
                "action_delay_prob": {
                    "init_range": [0.0, 0.05],
                    "limits": [0.0, 0.7],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "action_latency": {
                    "init_range": [0.0, 0.0],
                    "limits": [0, 60],
                    "delta": 0.1,
                    "delta_style": "additive"
                },
                "affine_action_scaling": {
                    "init_range": [0.0, 0.0],
                    "limits": [0.0, 4.0],
                    "delta": 0.0,
                    "delta_style": "additive"
                },
                "affine_action_additive": {
                    "init_range": [0.0, 0.04],
                    "limits": [0.0, 4.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "affine_action_white": {
                    "init_range": [0.0, 0.04],
                    "limits": [0.0, 4.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "affine_cube_pose_scaling": {
                    "init_range": [0.0, 0.0],
                    "limits": [0.0, 4.0],
                    "delta": 0.0,
                    "delta_style": "additive"
                },
                "affine_cube_pose_additive": {
                    "init_range": [0.0, 0.04],
                    "limits": [0.0, 4.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "affine_cube_pose_white": {
                    "init_range": [0.0, 0.04],
                    "limits": [0.0, 4.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "affine_dof_pos_scaling": {
                    "init_range": [0.0, 0.0],
                    "limits": [0.0, 4.0],
                    "delta": 0.0,
                    "delta_style": "additive"
                },
                "affine_dof_pos_additive": {
                    "init_range": [0.0, 0.04],
                    "limits": [0.0, 4.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "affine_dof_pos_white": {
                    "init_range": [0.0, 0.04],
                    "limits": [0.0, 4.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                },
                "rna_alpha": {
                    "init_range": [0.0, 0.0],
                    "limits": [0.0, 1.0],
                    "delta": 0.01,
                    "delta_style": "additive"
                }
            }
        }
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
        'num_position_iterations': 8,
        'num_velocity_iterations': 0,
        'contact_offset': 0.002,
        'rest_offset': 0.0,
        'bounce_threshold_velocity': 0.2,
        'max_depenetration_velocity': 1.0,
        'default_buffer_size_multiplier': 20.0,
        'max_gpu_contact_pairs': 8388608, # 1024*1024*8
        'contact_collection': 0
    },
}

env = AllegroHandDextremeADR(
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
cfg_ppo["learning_epochs"] = 5
cfg_ppo["mini_batches"] = 4  # 8 * 16384 / 32768
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 5e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
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


cfg_trainer = {"timesteps": 40000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()