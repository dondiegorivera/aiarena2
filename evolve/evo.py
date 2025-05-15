# evo_arena/evolve/evo.py
import numpy as np
# import random # No, use self.rng from np.random.default_rng
import os
# import shutil # Not directly used here anymore

from agents.brain import TinyNet
from arena.arena import Arena
from storage import persist
import config # Import config

class EvolutionOrchestrator:
    def __init__(self,
                 population_size=config.DEFAULT_POPULATION_SIZE,
                 num_elites=config.DEFAULT_NUM_ELITES,
                 initial_mutation_sigma=config.DEFAULT_MUTATION_SIGMA, # For initial genomes
                 # mutation_rate_genomes not used if sigma is self-adaptive and mutation always happens
                 # target_fitness_stdev not used with self-adaptive sigma as much

                 arena_width=config.ARENA_WIDTH, arena_height=config.ARENA_HEIGHT,
                 match_max_steps=config.MATCH_MAX_STEPS_TRAINING,
                 match_dt=config.SIMULATION_DT,
                 num_eval_matches_per_genome=config.DEFAULT_EVAL_MATCHES_PER_GENOME,
                 default_agent_hp=config.DEFAULT_AGENT_HP,
                 rng=None, # Expect an RNG instance
                 current_seed_for_saving=None # For logging in saved genome
                ):

        self.population_size = population_size
        self.num_elites = num_elites
        if not (0 <= self.num_elites <= self.population_size):
            raise ValueError("Number of elites must be between 0 and population size.")

        self.initial_mutation_sigma = initial_mutation_sigma
        # self.target_fitness_stdev = target_fitness_stdev # Diversity guard not in this plan

        self.arena_width = arena_width
        self.arena_height = arena_height
        self.match_max_steps = match_max_steps
        self.match_dt = match_dt
        self.num_eval_matches_per_genome = num_eval_matches_per_genome
        self.default_agent_hp = default_agent_hp

        self.population = [] # List of TinyNet instances
        self.generation = 0
        self.rng = rng if rng is not None else np.random.default_rng() # Fallback, but should be passed
        self.current_seed_for_saving = current_seed_for_saving

        # Arena for evaluations, pass RNG to it
        self.eval_arena = Arena(self.arena_width, self.arena_height,
                                wall_bounce_loss_factor=config.WALL_BOUNCE_LOSS_FACTOR, rng=self.rng)

        if not os.path.exists(config.BEST_GENOMES_PER_GENERATION_DIR):
            os.makedirs(config.BEST_GENOMES_PER_GENERATION_DIR)
        # No cleanup of old files by default, to preserve history across runs.

    def initialize_population(self):
        """Creates the initial population of random TinyNet brains."""
        self.population = []
        # Determine brain input/output sizes from config
        brain_input_size = config.BRAIN_INPUT_SIZE
        brain_output_size = config.BRAIN_OUTPUT_SIZE
        
        for _ in range(self.population_size):
            brain = TinyNet(
                input_size=brain_input_size,
                hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
                output_size=brain_output_size,
                rng=self.rng, # Pass RNG for weight initialization
                initial_sigma=self.initial_mutation_sigma # Initial sigma for this genome
            )
            # brain.fitness is already 0.0 by default in TinyNet
            self.population.append(brain)
        print(f"Initialized population with {self.population_size} random genomes.")

    @staticmethod
    def get_random_start_pose(arena_width, arena_height, agent_radius, rng_instance):
        """Generates a random valid starting pose for an agent."""
        clearance = 2.0 * agent_radius
        min_x, max_x = clearance, arena_width - clearance
        min_y, max_y = clearance, arena_height - clearance

        if min_x >= max_x or min_y >= max_y: # Arena too small for clearance
            # Fallback to center or raise error
            print(f"Warning: Arena too small for {clearance=}. Placing agent near center.")
            x = arena_width / 2
            y = arena_height / 2
        else:
            x = rng_instance.uniform(min_x, max_x)
            y = rng_instance.uniform(min_y, max_y)
        
        angle_deg = rng_instance.uniform(0, 360)
        return x, y, angle_deg

    def evaluate_population(self):
        """Evaluates the fitness of each genome in the current population."""
        if not self.population:
            print("Population is empty. Cannot evaluate.")
            return

        for genome_A in self.population:
            genome_A.fitness = 0.0 # Reset fitness for this generation's evaluation
            total_damage_dealt_by_A = 0.0
            total_ticks_survived_by_A = 0.0

            for _ in range(self.num_eval_matches_per_genome):
                # Select a random opponent (genome_B)
                if self.population_size > 1:
                    possible_opponents = [g for g in self.population if g is not genome_A]
                    if not possible_opponents: # Should only happen if pop_size is 1
                        genome_B = genome_A # Self-play if only one agent
                    else:
                        genome_B = self.rng.choice(possible_opponents)
                else: # Population size is 1
                    genome_B = genome_A # Self-play

                # Randomize starting positions for each match
                start_pos_A = EvolutionOrchestrator.get_random_start_pose(
                    self.arena_width, self.arena_height, config.AGENT_RADIUS, self.rng
                )
                start_pos_B = EvolutionOrchestrator.get_random_start_pose(
                    self.arena_width, self.arena_height, config.AGENT_RADIUS, self.rng
                )

                agent_configs = [
                    {'brain': genome_A, 'team_id': 1, 'agent_id': f"gA_gen{self.generation}",
                     'start_pos': start_pos_A, 'hp': self.default_agent_hp},
                    {'brain': genome_B, 'team_id': 2, 'agent_id': f"gB_gen{self.generation}",
                     'start_pos': start_pos_B, 'hp': self.default_agent_hp}
                ]
                
                match_results = self.eval_arena.run_match(agent_configs, self.match_max_steps, self.match_dt)
                
                # Extract results for genome_A (team_id 1)
                damage_A_this_match = 0
                # Ticks survived by A = match_results['duration_steps'] if A was alive at the end.
                # If A died, its survival ticks would be less. Arena.run_match needs to return this per agent.
                # For now, simplify: if A wins or draws, survived full duration. If lost, it died.
                # This is not perfectly accurate. The 'agents_final_state' has this info.

                hp_A_final = 0
                hp_B_final = 0
                is_A_alive_final = False

                for agent_state in match_results['agents_final_state']:
                    if agent_state['team_id'] == 1: # Genome A
                        damage_A_this_match = agent_state['damage_dealt']
                        hp_A_final = agent_state['hp']
                        is_A_alive_final = agent_state['is_alive']
                    elif agent_state['team_id'] == 2: # Genome B
                        hp_B_final = agent_state['hp']

                # Basic win/loss/draw score (can be more nuanced)
                # Based on existing main.py, +1 for win, -1 for loss, plus HP diff
                score_from_outcome = 0.0
                max_hp_norm = float(self.default_agent_hp)
                if match_results['winner_team_id'] == 1: # Genome A won
                    score_from_outcome = 1.0
                    if max_hp_norm > 0: score_from_outcome += (hp_A_final / max_hp_norm) * 0.5
                elif match_results['winner_team_id'] == 2: # Genome A lost
                    score_from_outcome = -1.0 # Negative score for loss
                    if max_hp_norm > 0: score_from_outcome -= ((max_hp_norm - hp_B_final) / max_hp_norm) * 0.5
                else: # Draw
                    score_from_outcome = 0.1 # Small positive for draw
                    if max_hp_norm > 0:
                        hp_diff_norm = (hp_A_final - hp_B_final) / max_hp_norm
                        score_from_outcome += hp_diff_norm * 0.25
                
                genome_A.fitness += score_from_outcome # Add win/loss based score

                total_damage_dealt_by_A += damage_A_this_match
                
                # Ticks survived: if A is alive at end, survived all steps.
                # If A died, need its death tick. Arena.run_match currently returns total duration.
                # For now, assume if A is alive, survived match_results['duration_steps'].
                # If A is not alive, means it died. How many ticks? This needs refinement in Arena/AgentBody state.
                # Simplified:
                if is_A_alive_final:
                    total_ticks_survived_by_A += match_results['duration_steps']
                else: # Died, approximate survival. This is a rough estimate.
                      # A better way is for Arena to report individual survival times.
                      # For now, let's say half duration if died. This is very coarse.
                    total_ticks_survived_by_A += match_results['duration_steps'] * (hp_A_final / max_hp_norm if max_hp_norm > 0 else 0.5)


            # Average base fitness over matches
            if self.num_eval_matches_per_genome > 0:
                genome_A.fitness /= self.num_eval_matches_per_genome
                avg_damage_dealt = total_damage_dealt_by_A / self.num_eval_matches_per_genome
                avg_ticks_survived = total_ticks_survived_by_A / self.num_eval_matches_per_genome
            else:
                avg_damage_dealt = 0
                avg_ticks_survived = 0

            # Add damage and survival bonuses
            genome_A.fitness += config.C_DAMAGE * avg_damage_dealt
            genome_A.fitness += config.C_SURVIVAL * avg_ticks_survived


    def select_and_reproduce(self):
        """Selects parents using tournament selection and creates a new population."""
        if not self.population:
            print("Population is empty. Cannot select and reproduce.")
            return

        new_population = []
        self.population.sort(key=lambda genome: genome.fitness, reverse=True) # For elites & tournament pool

        # 1. Elitism: Copy top N elites directly
        for i in range(self.num_elites):
            if i < len(self.population):
                # Elites are copied. They don't mutate their sigma here, that happens if they are selected as parents.
                # Or, elites could also undergo sigma mutation. For now, direct copy.
                elite_copy = TinyNet(w_in=self.population[i].w_in.copy(),
                                     w_out=self.population[i].w_out.copy(),
                                     input_size=self.population[i].input_size,
                                     hidden_size=self.population[i].hidden_size,
                                     output_size=self.population[i].output_size,
                                     rng=self.rng, # Give it the main RNG
                                     initial_sigma=self.population[i].mutation_sigma) # Preserve its sigma
                elite_copy.fitness = self.population[i].fitness # Preserve fitness
                new_population.append(elite_copy)


        # 2. Tournament Selection for the rest
        num_offspring_needed = self.population_size - len(new_population)
        
        parent_candidates = self.population # Can select from entire current population

        for _ in range(num_offspring_needed):
            # Tournament selection for parent(s)
            # EXPANDED_PLAN suggests "Mutation only (90%)" or "Crossover (10%)" from PROJECT.md's evolution loop.
            # Let's adopt this split for reproduction type.
            
            # Select parent 1 via tournament
            tournament_indices_p1 = self.rng.choice(len(parent_candidates), size=config.TOURNAMENT_SIZE_K, replace=False)
            tournament_competitors_p1 = [parent_candidates[i] for i in tournament_indices_p1]
            tournament_competitors_p1.sort(key=lambda g: g.fitness, reverse=True)
            parent1 = tournament_competitors_p1[0]

            if self.rng.random() < 0.10 and len(parent_candidates) > 1 : # Crossover (10% chance)
                # Select parent 2 via tournament (ensure different from P1 if possible)
                tournament_indices_p2 = self.rng.choice(len(parent_candidates), size=config.TOURNAMENT_SIZE_K, replace=False)
                tournament_competitors_p2 = [parent_candidates[i] for i in tournament_indices_p2]
                # Ensure p2 is different from p1 if population diversity allows
                # Simple way: re-pick if same and pool is large enough
                p2_candidate_pool = [g for g in tournament_competitors_p2 if g is not parent1]
                if p2_candidate_pool:
                    p2_candidate_pool.sort(key=lambda g: g.fitness, reverse=True)
                    parent2 = p2_candidate_pool[0]
                else: # All competitors were parent1 or list empty, just pick best from original tourney
                    tournament_competitors_p2.sort(key=lambda g: g.fitness, reverse=True)
                    parent2 = tournament_competitors_p2[0]
                
                child_genome = TinyNet.crossover(parent1, parent2, rng_instance=self.rng)
                child_genome = child_genome.mutate() # Mutate after crossover (uses child's sigma, inherited from P1)
            else: # Mutation only (90% chance)
                # Parent1 is mutated. Mutate method returns a new instance.
                child_genome = parent1.mutate() # This also mutates parent1's sigma for the child

            new_population.append(child_genome)

        self.population = new_population


    def run_evolution(self, num_generations):
        """Runs the main evolutionary loop."""
        print(f"Starting evolution for {num_generations} generations using dt={self.match_dt:.4f}.")
        if not self.population:
            self.initialize_population()

        all_time_best_fitness = -float('inf')
        all_time_best_genome_path = None

        for gen_idx in range(num_generations):
            self.generation = gen_idx
            print(f"\n--- Generation {self.generation}/{num_generations-1} ---")

            self.evaluate_population()
            print(f"Finished evaluating population for generation {self.generation}.")

            if not self.population:
                print("Error: Population became empty during evolution."); break

            self.population.sort(key=lambda genome: genome.fitness, reverse=True)

            if self.population:
                current_gen_best_genome = self.population[0]
                best_fitness = current_gen_best_genome.fitness
                avg_fitness = np.mean([g.fitness for g in self.population]) if self.population else 0.0
                avg_sigma = np.mean([g.mutation_sigma for g in self.population]) if self.population else 0.0
                print(f"Stats: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}, Avg Sigma = {avg_sigma:.5f}")

                try:
                    saved_path_gen_best = persist.save_genome(
                        current_gen_best_genome,
                        filename_prefix="gen_best",
                        directory=config.BEST_GENOMES_PER_GENERATION_DIR,
                        generation=self.generation,
                        fitness=current_gen_best_genome.fitness,
                        rng_seed_value=self.current_seed_for_saving # Log the overall run seed
                    )
                    # print(f"Saved generation {self.generation} best genome to: {saved_path_gen_best}")
                    if best_fitness > all_time_best_fitness:
                        all_time_best_fitness = best_fitness
                        all_time_best_genome_path = saved_path_gen_best
                except Exception as e:
                    print(f"Error saving generation {self.generation} best genome: {e}")
            else:
                 print("Warning: Population is empty after evaluation/sorting.")

            if gen_idx < num_generations - 1:
                self.select_and_reproduce()
                # print(f"Created new population for generation {self.generation + 1}.")

        print("\nEvolution finished.")
        print(f"All-time best fitness during this run: {all_time_best_fitness:.4f}")
        if all_time_best_genome_path:
            print(f"Path to a genome achieving this fitness: {all_time_best_genome_path}")
        return self.population