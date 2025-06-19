# satellite_config.py

from satellite.configs.base_config import BaseConfig

from pathlib import Path
import numpy as np

NUM_ENVS = 1024
N_EPOCHS = 1024
HEADLESS = True
FORCE_RENDER = False
PROFILE = False

class SatelliteConfig(BaseConfig):
    set_seed = False
    seed = 42

    physics_engine = 'physx'

    rl_device="cuda:0"
    sim_device="cuda:0"
    graphics_device_id=0
    headless=HEADLESS
    virtual_screen_capture=False
    force_render=FORCE_RENDER

    profile = PROFILE
    
    class env:  
        numEnvs = NUM_ENVS
   
        numObservations = 14 # [x,y,z,w, dx,dy,dz,dw, ax,ay,az, actX,actY,actZ]

        numStates = 17 # [x,y,z,w, dx,dy,dz,dw, ax,ay,az, actX,actY,actZ, vx,vy,vz]

        numActions = 3
        
        envSpacing = 4.0

        sensor_noise_std = 0.0
        actuation_noise_std = 0.0
        
        threshold_ang_goal = 0.01745        # soglia in radianti per orientamento
        threshold_vel_goal = 0.01745        # soglia in rad/sec per la differenza di velocità
        overspeed_ang_vel =  0.78540        # soglia in rad/sec per l'overspeed
        episode_length_s = 120              # soglia in secondi per la terminazione di una singola simulazione
        
        clipActions = np.Inf
        clipObservations = np.Inf

        torque_scale = 10

        class asset:
            assetRoot = str(Path(__file__).resolve().parent.parent)
            assetFileName = "satellite.urdf"
            assetName = "satellite"

            init_pos_p = [0, 0, 0]    # posizione iniziale del satellite [x,y,z]
            init_pos_r = [0, 0, 0, 1] # attitude iniziale del satellite [x,y,z,w]

            #disable_gravity
            #collapse_fixed_joints
            #slices_per_cylinder
            #replace_cylinder_with_capsule
            #fix_base_link
            #default_dof_drive_mode
            #self_collisions
            #flip_visual_attachments

            #density
            #angular_damping
            #linear_damping
            #max_angular_velocity
            #max_linear_velocity
            #armature
            #thickness

    class sim:
        dt = 1.0 / 60.0
        gravity = [0.0, 0.0, 0.0] # [m/s^2]
        up_axis = 'z'
        use_gpu_pipeline = True
        substeps = 2
        
        #num_client_threads
        #stress_visualization
        #stress_visualization_max
        #stress_visualization_min
        
        class physx:
            use_gpu = True
            solver_type = 1
            num_threads = 4
            num_position_iterations = 4
            num_velocity_iterations = 1
            #contact_offset
            #rest_offset
            #bounce_threshold_velocity
            #contact_collection
            #default_buffer_size_multiplier
            #max_depenetration_velocity
            #max_gpu_contact_pairs
            #num_subscenes
            #always_use_articulations
            #friction_correlation_distance
            #friction_offset_threshold
            
        #class flex:
            #solver_type
            #num_outer_iterations
            #num_inner_iterations
            #relaxation
            #warm_start
            #contact_regularization
            #deterministic_mode
            #dynamic_friction
            #friction_mode
            #geometric_stiffness
            #max_rigid_contacts
            #max_soft_contacts
            #particle_friction
            #return_contacts
            #shape_collision_distance
            #shape_collision_margin
            #static_friction

    class task:
        randomize = False

    class rl:
        class PPO:
            num_envs = NUM_ENVS
            rollouts = 16
            learning_epochs = 8
            mini_batches = 1
            discount_factor = 0.99
            lambda_ = 0.95
            learning_rate = 3e-4
            grad_norm_clip = 1.0
            ratio_clip = 0.2
            value_clip = 0.2
            clip_predicted_values = True
            entropy_loss_scale = 0.00
            value_loss_scale = 2.0
            kl_threshold = 0
            random_timesteps = 0
            learning_starts = 0
            
            class experiment:
                    write_interval = 10
                    checkpoint_interval = 100
                    directory = "./runs/satellite"
                    wandb = False

        class trainer:
            rollouts = 16
            n_epochs = N_EPOCHS
            timesteps = rollouts * n_epochs
            disable_progressbar = False
            headless = HEADLESS

        class memory:
            rollouts = 16

    class pid:
        class rate:
            kp = 0.5
            ki = 0.0
            kd = 0.1
    
    class controller:
        controller_logic = False