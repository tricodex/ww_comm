# settings

env_kwargs = {
    "n_pursuers": 6,  # number of pursuing archea (agents)
    "n_evaders": 4,  # number of food objects
    "n_poisons": 10,  # number of poison objects
    "n_coop": 2,  # number of pursuing archea (agents) that must be touching food at the same time to consume it
    "n_sensors": 30,  # number of sensors on all pursuing archea (agents)
    "sensor_range": 0.2,  # length of sensor dendrite on all pursuing archea (agents)
    "radius": 0.015,  # archea base radius. Pursuer: radius, food: 2 x radius, poison: 3/4 x radius
    "obstacle_radius": 0.05,  # radius of obstacle object
    "n_obstacles": 1,  # number of obstacle objects
    "obstacle_coord": [(0.5, 0.5)], #, (0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)],  # coordinate of obstacle objects
    "pursuer_max_accel": 0.01,  # pursuer archea maximum acceleration (maximum action size)
    "evader_speed": 0.01,  # food speed
    "poison_speed": 0.01,  # poison speed
    "poison_reward": -1.0,  # reward for pursuer consuming a poison object (typically negative)
    "food_reward": 10.0,  # reward for pursuers consuming a food object
    "encounter_reward": 0.01,  # reward for a pursuer colliding with a food object
    "thrust_penalty": -0.5,  # scaling factor for the negative reward used to penalize large actions
    "local_ratio": 0.5,  # Proportion of reward allocated locally vs distributed globally among all agents (vs 1.0)
    "speed_features": True,  # toggles whether pursuing archea (agent) sensors detect speed of other objects and archea
    "max_cycles": 9830  # After max_cycles steps all agents will return done; maybe 98304?
}

# Hyperparameter space for PPO
hyperparam_space_ppo = {
            'learning_rate': [1e-4, 1e-3, 1e-5, 1e-6],
            'batch_size': [64, 128, 256, 512, 1024],
            'gamma': [0.8, 0.925, 0.95, 0.975, 0.999],
            'gae_lambda': [0.8, 0.9, 0.95],
            'n_steps': [1024, 2048, 4096, 8192, 16384], 
            'ent_coef': [0.0, 0.001, 0.00001],
            'vf_coef': [0.25, 0.5, 1.0],
            'max_grad_norm': [1.0, 5.0, 10.0],
            'clip_range': [0.1, 0.5, 0.9],
            
        }

# Hyperparameter space for SAC
hyperparam_space_sac = {
    'learning_rate': [1e-4, 1e-3, 1e-5, 1e-6],
    'batch_size': [64, 128, 256, 512, 1024],
    'gamma': [0.8, 0.925, 0.95, 0.975, 0.999],
    'tau': [0.005, 0.01, 0.02],
    'ent_coef': ['auto', 0.1, 0.01],
    'target_entropy': ['auto', 0.0, -1.0, -0.5],
    'use_sde': [True, False],
    'sde_sample_freq': [1, -1, 20],
    'learning_starts': [1000, 5000, 10000],
    'buffer_size': [10000, 50000, 100000] 
}


# Training and evaluation settings
            
EPISODES = 10                  # Number of episodes for training
EPISODE_LENGTH = 98304           # Length of each episode in steps
EVAL_NUM_GAMES = 10              # Number of games for model evaluation

# Genetic Algorithm Optimization settings
GA_POPULATION_SIZE = 30          # Population size for genetic algorithm
GA_GENERATIONS = 20              # Number of generations for genetic algorithm
GA_ELITISM_SIZE = 2  