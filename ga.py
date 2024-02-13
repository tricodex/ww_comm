# ga.py

import random
import logging
import matplotlib.pyplot as plt
from settings import hyperparam_space_ppo, hyperparam_space_sac, env_kwargs
import datetime
import os
import concurrent.futures


OPTIMIZE_DIR = 'optimize'

class GeneticHyperparamOptimizer:
    def __init__(self, model_name):
        logging.basicConfig(filename='genetic_algo.log', level=logging.INFO)
        self.model_name = model_name

        # Add current datetime to logging
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Current datetime: {current_datetime}")

        # Select hyperparameter space based on model name
        if model_name == "PPO":
            self.hyperparam_space = hyperparam_space_ppo
        elif model_name == "SAC":
            self.hyperparam_space = hyperparam_space_sac
        else:
            raise ValueError("Invalid model name")

    def generate_individual(self):
        """
        Create an individual with random hyperparameters.
        """
        return {k: random.choice(v) for k, v in self.hyperparam_space.items()}

    
    
    def mutate(self, individual):
        """
        Mutate multiple hyperparameters of an individual.
        """
        num_mutations = random.randint(1, len(self.hyperparam_space))  # Number of hyperparameters to mutate
        mutation_keys = random.sample(list(individual.keys()), num_mutations)

        for mutation_key in mutation_keys:
            current_value = individual[mutation_key]
            value_range = self.hyperparam_space[mutation_key]

            if all(isinstance(x, (int, float)) for x in value_range):
                # Handle numeric values
                mutation_range = max(value_range) - min(value_range)
                new_value = current_value + random.uniform(-mutation_range * 0.1, mutation_range * 0.1)
                individual[mutation_key] = max(min(new_value, max(value_range)), min(value_range))
            elif all(isinstance(x, str) for x in value_range):
                # Handle string values
                individual[mutation_key] = random.choice(value_range)
            elif isinstance(current_value, str):
                # Special handling for mixed types, focusing on strings
                individual[mutation_key] = random.choice([x for x in value_range if isinstance(x, str)])
                # Handling for boolean types
            elif isinstance(current_value, bool):
                individual[mutation_key] = not current_value  # Toggle boolean value

            else:
                # Special handling for mixed types, focusing on numbers
                numeric_values = [x for x in value_range if isinstance(x, (int, float))]
                mutation_range = max(numeric_values) - min(numeric_values)
                new_value = current_value + random.uniform(-mutation_range * 0.1, mutation_range * 0.1)
                individual[mutation_key] = max(min(new_value, max(numeric_values)), min(numeric_values))        
        return individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two individuals with fitness consideration.
        """
        child = {}
        for key in self.hyperparam_space.keys():
            if key in ["learning_rate", "gamma"]:  # For sensitive parameters, favor the better parent
                child[key] = parent1[key] if parent1["fitness"] > parent2["fitness"] else parent2[key]
            else:
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child

    def evaluate(self, individual, train_function, eval_function, env_fn):
        # Select hyperparameters based on the model
        if self.model_name == "PPO":
            # Filter out only those hyperparameters that are relevant for PPO
            ppo_params = {k: individual[k] for k in individual if k in hyperparam_space_ppo}
            hyperparams = ppo_params
            if 'buffer_size' in hyperparams:
                hyperparams['buffer_size'] = int(hyperparams['buffer_size'])
            if 'n_steps' in hyperparams:
                hyperparams['n_steps'] = int(hyperparams['n_steps'])
            if 'batch_size' in hyperparams:
                hyperparams['batch_size'] = int(hyperparams['batch_size'])
        elif self.model_name == "SAC":
            # Filter out only those hyperparameters that are relevant for SAC
            sac_params = {k: individual[k] for k in individual if k in hyperparam_space_sac}
            hyperparams = sac_params
            # Ensure buffer_size is an integer
            hyperparams['buffer_size'] = int(hyperparams['buffer_size'])
        else:
            raise ValueError("Invalid model name")
        
        # Pass the relevant hyperparameters to the train function
        train_function(env_fn, self.model_name, OPTIMIZE_DIR, steps=196_608, seed=0, **hyperparams)

        # Evaluate the trained model
        avg_reward = eval_function(env_fn, self.model_name, model_subdir=OPTIMIZE_DIR, num_games=2 )
        logging.info(f"Evaluating Individual: {individual}, Avg Reward: {avg_reward}")
        # Add fitness score to the individual
        individual['fitness'] = avg_reward

        return avg_reward


    def run(self, train_function, eval_function, env_fn, population_size=10, generations=5, elitism_size=2):
        """
        Run the genetic algorithm with elitism.
        """
        population = [self.generate_individual() for _ in range(population_size)]
        print(f"Initial population: {population}")
        best_scores = []
        
        for generation in range(generations):
            print(f"Generation {generation + 1} of {generations}")
            # Evaluate individuals and store fitness scores separately
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate(individual, train_function, eval_function, env_fn)
                fitness_scores.append(fitness)

            # Sort individuals based on fitness
            sorted_pairs = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            sorted_population = [pair[1] for pair in sorted_pairs]

            # Elitism - carry over some best individuals
            next_generation = sorted_population[:elitism_size]
            
            # Crossover and mutation for the rest
            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(sorted_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation
            best_score = max(fitness_scores)
            best_scores.append(best_score)
            logging.info(f"Generation {generation + 1}, Best Score: {best_score}")
        
        self.plot_performance(best_scores)
        return sorted_population[0]
    
    def run_parallel(self, train_function, eval_function, env_fn, population_size=10, generations=5, elitism_size=2):
        population = [self.generate_individual() for _ in range(population_size)]
        
        for generation in range(generations):
            print(f"Starting Generation {generation + 1}")

            # Evaluate all individuals in the population in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for individual in population:
                    run_id = f"gen{generation}_indiv{population.index(individual)}"
                    future = executor.submit(self.evaluate_parallel, individual, train_function, eval_function, env_fn, run_id)
                    futures.append(future)

                results = [future.result() for future in futures]

            # Associate each individual with its fitness score
            for individual, fitness in zip(population, results):
                individual['fitness'] = fitness

            # Sort the population based on fitness in descending order
            population.sort(key=lambda ind: ind['fitness'], reverse=True)

            # Elitism - carry over some best individuals to the next generation
            next_generation = population[:elitism_size]

            # Crossover and mutation for the rest of the next generation
            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation
            print(f"Generation {generation + 1} Complete. Best Score: {population[0]['fitness']}")

        return population[0]  # Return the best individual
    
    def evaluate_parallel(self, individual, train_function, eval_function, env_fn, run_id):
        # Select hyperparameters based on the model
        if self.model_name == "PPO":
            # Filter out only those hyperparameters that are relevant for PPO
            ppo_params = {k: individual[k] for k in individual if k in hyperparam_space_ppo}
            hyperparams = ppo_params
            if 'buffer_size' in hyperparams:
                hyperparams['buffer_size'] = int(hyperparams['buffer_size'])
            if 'n_steps' in hyperparams:
                hyperparams['n_steps'] = int(hyperparams['n_steps'])
            if 'batch_size' in hyperparams:
                hyperparams['batch_size'] = int(hyperparams['batch_size'])
        elif self.model_name == "SAC":
            # Filter out only those hyperparameters that are relevant for SAC
            sac_params = {k: individual[k] for k in individual if k in hyperparam_space_sac}
            hyperparams = sac_params
            hyperparams['buffer_size'] = int(hyperparams['buffer_size'])
        else:
            raise ValueError("Invalid model name")

        # Pass the relevant hyperparameters to the train function, including run_id
        train_function(env_fn, self.model_name, OPTIMIZE_DIR, run_id, steps=196_608, seed=0, **hyperparams)

        # Evaluate the trained model
        avg_reward = eval_function(env_fn, self.model_name, model_subdir=OPTIMIZE_DIR, run_id=run_id, num_games=10)
        logging.info(f"[{run_id}] Evaluating Individual: {individual}, Avg Reward: {avg_reward}")

        return avg_reward



    def plot_performance(self, best_scores):
        plt.plot(best_scores)
        plt.xlabel('Generation')
        plt.ylabel('Best Score')
        plt.title('Best Score Evolution')

        # Directory where plots will be saved
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Formatting the filename with the current date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plot_filename = os.path.join(plots_dir, f'performance_plot_{current_time}.png')

        plt.savefig(plot_filename)
        # plt.show()  
