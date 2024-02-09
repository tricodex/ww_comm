# heuristic_policy.py

import numpy as np

def simple_policy(observation, n_sensors, sensor_range):
    """
    A heuristic policy for the Waterworld simulation environment.

    This policy guides an agent based on sensor readings for food and poison.
    The agent's behavior is determined by the proximity of these objects,
    as indicated by the sensor readings.
    
    Behavior:
    1. The policy first checks the distance of food and poison from the agent,
       using the sensor readings provided in the observation.

    2. If food is detected within a threshold distance (set to the sensor range),
       the agent moves towards the closest piece of food. The direction is 
       determined by identifying the sensor with the minimum distance reading 
       to the food.

    3. If poison is detected within the same threshold distance and no food is 
       detected, the agent moves away from the closest piece of poison. Again,
       the direction is determined by the sensor that detects the closest poison.

    4. If neither food nor poison is detected within the threshold, the agent 
       moves in a random direction.


    """

    # Extract sensor readings for food and poison
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]

    # Threshold to consider an object "detected"
    detection_threshold = 1 * sensor_range

    # Initialize action
    action = np.array([0.0, 0.0])

    # Check for food and poison
    if np.any(food_dist < detection_threshold):
        # Move towards the closest food
        closest_food_index = np.argmin(food_dist)
        action[0] = np.cos(closest_food_index * 2 * np.pi / n_sensors)
        action[1] = np.sin(closest_food_index * 2 * np.pi / n_sensors)
    elif np.any(poison_dist < detection_threshold):
        # Move away from the closest poison
        closest_poison_index = np.argmin(poison_dist)
        action[0] = -np.cos(closest_poison_index * 2 * np.pi / n_sensors)
        action[1] = -np.sin(closest_poison_index * 2 * np.pi / n_sensors)
    else:
        # Random wander
        random_direction = np.random.rand() * 2 * np.pi
        action[0] = np.cos(random_direction)
        action[1] = np.sin(random_direction)

    # Assuming 'action' is the variable holding your calculated action
    action = np.array(action, dtype=np.float32)


    return action

