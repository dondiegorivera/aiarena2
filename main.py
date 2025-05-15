# evo_arena/main.py
import pygame
import argparse
import os
import sys
import glob
import itertools
# import shutil # shutil was in original main.py but not used in its current form after Evo Orchestrator handles dir.

import numpy as np # For GD/RL. Keep for now.

# Custom modules
import config # ✅ USE CONFIG
import rand   # ✅ USE RAND FOR RNG

from arena.arena import Arena
from agents.body import AgentBody
from agents.brain import TinyNet
from ui.viewer import Viewer
from evolve.evo import EvolutionOrchestrator
from storage import persist


# --- Core Simulation Functions ---
def run_manual_simulation(opponent_genome_path=None):
    """Runs a manual simulation with player control against an optional AI opponent."""
    rng = rand.get_rng() # Get the global RNG
    game_arena = Arena(config.ARENA_WIDTH, config.ARENA_HEIGHT, wall_bounce_loss_factor=config.WALL_BOUNCE_LOSS_FACTOR, rng=rng)
    manual_agent = AgentBody(
        x=config.ARENA_WIDTH / 2, y=config.ARENA_HEIGHT - 100, angle_deg=-90,
        base_speed=config.AGENT_BASE_SPEED, rotation_speed_dps=config.AGENT_ROTATION_SPEED_DPS, radius=config.AGENT_RADIUS,
        color=config.MANUAL_AGENT_COLOR, agent_id="player", team_id=config.PLAYER_TEAM_ID,
        hp=config.DEFAULT_AGENT_HP + 50, brain=None, # Player gets more HP
        weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
        weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
        cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR,
        rng=rng, # Pass RNG
        # RNN state not needed for manual/None brain
    )
    game_arena.add_agent(manual_agent)

    ai_opponent_brain = None
    ai_opponent_id = "ai_opponent_random"
    brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
    brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)

    if opponent_genome_path:
        print(f"Attempting to load opponent genome from: {opponent_genome_path}")
        try:
            # Pass RNG to load_genome if it needs to init sigma or other RNG-dependent params,
            # though typically load_genome just loads weights.
            ai_opponent_brain = persist.load_genome(
                opponent_genome_path,
                input_size=brain_input_size,
                hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
                output_size=brain_output_size,
                rng=rng # For initializing sigma if not in file
            )
            ai_opponent_id = f"ai_trained_{os.path.basename(opponent_genome_path).split('.')[0]}"
            print(f"Successfully loaded trained opponent: {ai_opponent_id}")
        except FileNotFoundError:
            print(f"Warning: Opponent genome file not found at {opponent_genome_path}. Using random AI opponent.")
            ai_opponent_brain = TinyNet(
                input_size=brain_input_size,
                hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
                output_size=brain_output_size,
                rng=rng
            )
        except Exception as e:
            print(f"Warning: Error loading opponent genome ({e}). Using random AI opponent.")
            ai_opponent_brain = TinyNet(
                input_size=brain_input_size,
                hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
                output_size=brain_output_size,
                rng=rng
            )
    else:
        print("No opponent genome specified. Using random AI opponent.")
        ai_opponent_brain = TinyNet(
            input_size=brain_input_size,
            hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
            output_size=brain_output_size,
            rng=rng
        )

    ai_opponent = AgentBody(
        x=config.ARENA_WIDTH / 2, y=100, angle_deg=90,
        base_speed=config.AGENT_BASE_SPEED * 0.9, rotation_speed_dps=config.AGENT_ROTATION_SPEED_DPS * 0.9,
        radius=config.AGENT_RADIUS, color=config.AI_OPPONENT_COLOR, agent_id=ai_opponent_id,
        team_id=config.AI_OPPONENT_TEAM_ID, hp=config.DEFAULT_AGENT_HP, brain=ai_opponent_brain,
        weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
        weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
        cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR,
        rng=rng, # Pass RNG
        use_rnn=config.USE_RNN, rnn_hidden_size=config.RNN_HIDDEN_SIZE
    )
    game_arena.add_agent(ai_opponent)

    if not opponent_genome_path : # Add dummies if no specific opponent loaded
        dummy_target_1 = AgentBody(
            x=100, y=config.ARENA_HEIGHT / 2, angle_deg=0, base_speed=0,
            rotation_speed_dps=0, radius=config.AGENT_RADIUS, color=config.DUMMY_AGENT_COLOR,
            agent_id="dummy_1", team_id=config.DUMMY_TEAM_ID, hp=100, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0,
            rng=rng # Dummies also get RNG for consistency if they ever need it
        )
        game_arena.add_agent(dummy_target_1)
        dummy_target_2 = AgentBody(
            x=config.ARENA_WIDTH - 100, y=config.ARENA_HEIGHT / 2, angle_deg=180, base_speed=0,
            rotation_speed_dps=0, radius=config.AGENT_RADIUS, color=config.DUMMY_AGENT_COLOR,
            agent_id="dummy_2", team_id=config.DUMMY_TEAM_ID, hp=100, is_dummy=True, brain=None,
            weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0,
            rng=rng
        )
        game_arena.add_agent(dummy_target_2)

    title = f"Manual Play vs {ai_opponent_id}"
    # Pass RNG to Viewer if it needs it (e.g. for visual effects), currently not needed
    game_viewer = Viewer(config.ARENA_WIDTH, config.ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(config.VISUAL_FPS, manual_agent_id="player")


def run_training_session(generations, population_size, num_elites, mutation_sigma, eval_matches, match_steps, sim_dt, current_seed):
    """Runs an evolutionary training session."""
    rng = rand.get_rng() # Get the global RNG
    print("\n" + "="*30); print(" STARTING EVOLUTIONARY TRAINING "); print("="*30)
    print(f"Seed for this run: {current_seed}") # ✅ Log reproducibility info
    print(f"Generations={generations}, Population={population_size}, Elites={num_elites}")
    print(f"Initial Mutation Sigma={mutation_sigma} (self-adaptive)")
    print(f"Eval Matches per Genome={eval_matches}, Tournament Size K={config.TOURNAMENT_SIZE_K}")
    print(f"Sim DT: {sim_dt:.4f} ({1.0/sim_dt:.1f} tps), Match Steps: {match_steps} (~{match_steps*sim_dt:.1f}s)")
    print(f"Agent HP for Eval: {config.DEFAULT_AGENT_HP}")
    print(f"Lidar Rays: {config.LIDAR_NUM_RAYS}, RNN: {config.USE_RNN} (Hidden: {config.RNN_HIDDEN_SIZE if config.USE_RNN else 'N/A'})")
    print(f"Fitness: Damage Coeff={config.C_DAMAGE}, Survival Coeff={config.C_SURVIVAL}")
    print(f"Best genomes saved to: {config.BEST_GENOMES_PER_GENERATION_DIR}")
    print("="*30 + "\n")

    evo_orchestrator = EvolutionOrchestrator(
        population_size=population_size,
        num_elites=num_elites,
        initial_mutation_sigma=mutation_sigma, # For initial sigma of genomes
        arena_width=config.ARENA_WIDTH,
        arena_height=config.ARENA_HEIGHT,
        match_max_steps=match_steps,
        match_dt=sim_dt,
        num_eval_matches_per_genome=eval_matches,
        default_agent_hp=config.DEFAULT_AGENT_HP,
        rng=rng, # Pass RNG
        current_seed_for_saving=current_seed # Pass seed for metadata
    )
    final_population = evo_orchestrator.run_evolution(num_generations=generations)

    if final_population:
        final_population.sort(key=lambda g: g.fitness, reverse=True)
        best_overall = final_population[0]
        print(f"\nTraining complete. Best overall fitness in final population: {best_overall.fitness:.4f}")
        final_best_dir = os.path.join(config.GENOME_STORAGE_DIR, "final_bests")
        if not os.path.exists(final_best_dir): os.makedirs(final_best_dir)
        # Pass RNG for saving, though persist.save_genome itself doesn't use RNG, it logs the seed.
        saved_path = persist.save_genome(
            best_overall, "final_best_genome", final_best_dir,
            generations, best_overall.fitness, rng_seed_value=current_seed
        )
        print(f"Saved best overall genome from final population to: {saved_path}")
    else:
        print("Training completed, but no final population data available.")
    print("="*30 + "\n")


def run_visual_match(genome_path1, genome_path2, num_games=1):
    """Runs a visual match series between two AI agents."""
    rng = rand.get_rng()
    ai1_name = os.path.basename(genome_path1).split('.')[0]
    ai2_name = os.path.basename(genome_path2).split('.')[0]

    print(f"\nRunning Visual Match Series (Best of {num_games}):")
    print(f"  AI 1 (Red): {ai1_name}")
    print(f"  AI 2 (Green): {ai2_name}")

    brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
    brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)

    try:
        brain1 = persist.load_genome(genome_path1, brain_input_size, config.BRAIN_HIDDEN_LAYER_SIZE, brain_output_size, rng=rng)
        brain2 = persist.load_genome(genome_path2, brain_input_size, config.BRAIN_HIDDEN_LAYER_SIZE, brain_output_size, rng=rng)
    except Exception as e:
        print(f"Error loading genome(s): {e}"); return

    score_ai1 = 0
    score_ai2 = 0

    game_arena = Arena(config.ARENA_WIDTH, config.ARENA_HEIGHT, wall_bounce_loss_factor=config.WALL_BOUNCE_LOSS_FACTOR, rng=rng)

    agent1_body = AgentBody(
        x=150, y=config.ARENA_HEIGHT / 2, angle_deg=0, base_speed=config.AGENT_BASE_SPEED,
        rotation_speed_dps=config.AGENT_ROTATION_SPEED_DPS, radius=config.AGENT_RADIUS,
        color=config.AI_OPPONENT_COLOR, agent_id=ai1_name, team_id=1, hp=config.DEFAULT_AGENT_HP, brain=brain1,
        weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
        weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
        cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR, rng=rng,
        use_rnn=config.USE_RNN, rnn_hidden_size=config.RNN_HIDDEN_SIZE
    )
    agent2_body = AgentBody(
        x=config.ARENA_WIDTH - 150, y=config.ARENA_HEIGHT / 2, angle_deg=180, base_speed=config.AGENT_BASE_SPEED,
        rotation_speed_dps=config.AGENT_ROTATION_SPEED_DPS, radius=config.AGENT_RADIUS,
        color=config.AI_AGENT_COLOR_2, agent_id=ai2_name, team_id=2, hp=config.DEFAULT_AGENT_HP, brain=brain2,
        weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
        weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
        cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR, rng=rng,
        use_rnn=config.USE_RNN, rnn_hidden_size=config.RNN_HIDDEN_SIZE
    )
    game_arena.add_agent(agent1_body)
    game_arena.add_agent(agent2_body)

    initial_viewer_title = f"Game 1/{num_games} | {ai1_name}: 0 - {ai2_name}: 0"
    game_viewer = Viewer(config.ARENA_WIDTH, config.ARENA_HEIGHT, game_arena, title=initial_viewer_title)

    for game_num in range(1, num_games + 1):
        print(f"\n--- Starting Game {game_num} of {num_games} ---")

        game_arena.reset_arena_and_agents() # Resets agents to their initial_x, initial_y etc.
        # For visual matches, we want fixed start positions, so re-apply them after reset.
        agent1_body.reset_state(x=150, y=config.ARENA_HEIGHT / 2, angle_deg=0, hp=config.DEFAULT_AGENT_HP)
        agent2_body.reset_state(x=config.ARENA_WIDTH - 150, y=config.ARENA_HEIGHT / 2, angle_deg=180, hp=config.DEFAULT_AGENT_HP)
        # Ensure brains are still assigned (reset_state doesn't touch brain)
        agent1_body.brain = brain1
        agent2_body.brain = brain2


        current_game_title = f"Game {game_num}/{num_games} | {ai1_name} (R): {score_ai1} - {ai2_name} (G): {score_ai2}"
        pygame.display.set_caption(current_game_title)

        running_this_game = True
        winner_message_this_game = "Game in progress..."

        match_visual_dt = 1.0 / config.VISUAL_FPS
        game_max_steps = int(config.MATCH_DURATION_SECONDS / match_visual_dt)

        for step in range(game_max_steps):
            if not pygame.display.get_init():
                print("Match series aborted (window closed).")
                return

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_this_game = False; winner_message_this_game = "Series Aborted (Quit)"
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_this_game = False; winner_message_this_game = "Game Skipped by User"
                        break
            if not running_this_game:
                break

            game_arena.update(match_visual_dt)
            game_viewer.render_frame(show_game_over_message=False) # Let logic below handle game over text

            is_over, winner_team_id, message = game_arena.check_match_end_conditions(max_duration_seconds=config.MATCH_DURATION_SECONDS)
            if is_over:
                winner_message_this_game = message
                if winner_team_id == 1:
                    score_ai1 += 1
                    winner_message_this_game += f" Winner: {ai1_name} (Red)"
                elif winner_team_id == 2:
                    score_ai2 += 1
                    winner_message_this_game += f" Winner: {ai2_name} (Green)"
                else:
                    winner_message_this_game += " (Draw)"
                break

            # Display current game time and HP on screen via Viewer
            hp_texts = [
                (f"{ai1_name} (Red) HP: {agent1_body.hp:.0f}", config.AI_OPPONENT_COLOR),
                (f"{ai2_name} (Green) HP: {agent2_body.hp:.0f}", config.AI_AGENT_COLOR_2)
            ]
            game_viewer.render_hud_texts(hp_texts=hp_texts)

            pygame.display.flip()
            game_viewer.clock.tick(config.VISUAL_FPS)

        print(f"Game {game_num} Result: {winner_message_this_game}")
        current_series_score_str = f"{ai1_name} (Red): {score_ai1}  -  {ai2_name} (Green): {score_ai2}"
        print(f"Current Series Score: {current_series_score_str}")
        pygame.display.set_caption(f"Game {game_num} Over! {current_series_score_str} | SPACE for Next")

        if not pygame.display.get_init() or "Series Aborted" in winner_message_this_game:
            if pygame.display.get_init(): pygame.quit()
            return

        if game_num < num_games:
            game_viewer.wait_for_keypress_to_continue(
                main_message=winner_message_this_game,
                sub_message=current_series_score_str,
                prompt_message="Press SPACE for Next Game or ESC to End Series",
                continue_key=pygame.K_SPACE,
                abort_key=pygame.K_ESCAPE
            )
            if not game_viewer.is_running(): # Check if aborted during wait
                 if pygame.display.get_init(): pygame.quit()
                 return

        if not pygame.display.get_init(): return

    print(f"\n--- Match Series Finished ---")
    final_score_str = f"FINAL SCORE: {ai1_name} (Red) {score_ai1}  -  {score_ai2} {ai2_name} (Green)"
    print(final_score_str)
    overall_winner_str = ""
    if score_ai1 > score_ai2: overall_winner_str = f"Overall Winner: {ai1_name} (Red)"
    elif score_ai2 > score_ai1: overall_winner_str = f"Overall Winner: {ai2_name} (Green)"
    else: overall_winner_str = "Overall Series is a Draw!"
    print(overall_winner_str)

    if pygame.display.get_init():
        pygame.display.set_caption(f"{final_score_str} | {overall_winner_str} | Press ESC")
        game_viewer.wait_for_keypress_to_continue(
            main_message=final_score_str,
            sub_message=overall_winner_str,
            prompt_message="Press ESC to return to menu.",
            continue_key=None, # No continue, only abort
            abort_key=pygame.K_ESCAPE,
            is_final_screen=True
        )
        if pygame.display.get_init(): pygame.quit()


def run_show_genome(genome_path, scenario='vs_dummies'):
    """Shows a single AI genome playing in a scenario."""
    rng = rand.get_rng()
    print(f"\nShowing genome: {os.path.basename(genome_path)} in scenario: {scenario}")

    brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
    brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)

    try:
        ai_brain = persist.load_genome(genome_path, brain_input_size, config.BRAIN_HIDDEN_LAYER_SIZE, brain_output_size, rng=rng)
    except Exception as e:
        print(f"Error loading genome: {e}"); return

    game_arena = Arena(config.ARENA_WIDTH, config.ARENA_HEIGHT, wall_bounce_loss_factor=config.WALL_BOUNCE_LOSS_FACTOR, rng=rng)
    ai_agent_show = AgentBody(
        x=config.ARENA_WIDTH / 2, y=config.ARENA_HEIGHT - 150, angle_deg=-90,
        base_speed=config.AGENT_BASE_SPEED, rotation_speed_dps=config.AGENT_ROTATION_SPEED_DPS,
        radius=config.AGENT_RADIUS, color=config.AI_OPPONENT_COLOR, agent_id=os.path.basename(genome_path).split('.')[0], team_id=1,
        hp=config.DEFAULT_AGENT_HP, brain=ai_brain,
        weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
        weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
        cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR, rng=rng,
        use_rnn=config.USE_RNN, rnn_hidden_size=config.RNN_HIDDEN_SIZE
    )
    game_arena.add_agent(ai_agent_show)

    if scenario == 'vs_dummies':
        game_arena.add_agent(AgentBody(x=config.ARENA_WIDTH/2, y=150, angle_deg=90, agent_id="dummy1", team_id=2, hp=150, is_dummy=True, radius=config.AGENT_RADIUS+5, color=config.DUMMY_AGENT_COLOR, base_speed=0, rotation_speed_dps=0, weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0, rng=rng))
        game_arena.add_agent(AgentBody(x=config.ARENA_WIDTH/4, y=config.ARENA_HEIGHT/2, angle_deg=0, agent_id="dummy2", team_id=2, hp=100, is_dummy=True, radius=config.AGENT_RADIUS+5, color=config.DUMMY_AGENT_COLOR, base_speed=0, rotation_speed_dps=0, weapon_range=0, weapon_arc_deg=0, weapon_cooldown_time=999, weapon_damage=0, rng=rng))

    title = f"Showcase: {os.path.basename(genome_path).split('.')[0]} ({scenario})"
    game_viewer = Viewer(config.ARENA_WIDTH, config.ARENA_HEIGHT, game_arena, title=title)
    game_viewer.run_simulation_loop(config.VISUAL_FPS)


def run_post_training_tournament(tournament_genome_dir=None, visual=False):
    """Runs a tournament between champion genomes."""
    rng = rand.get_rng()
    if tournament_genome_dir is None: tournament_genome_dir = config.BEST_GENOMES_PER_GENERATION_DIR
    print(f"\n--- Post-Training Tournament of Champions from '{tournament_genome_dir}' ---")
    if not os.path.exists(tournament_genome_dir):
        print(f"Dir not found: {tournament_genome_dir}. Run training first or specify dir."); return

    genome_files = glob.glob(os.path.join(tournament_genome_dir, "*.npz"))
    if len(genome_files) < 2: print(f"Need at least 2 genomes. Found {len(genome_files)}."); return
    print(f"Found {len(genome_files)} champion genomes.")

    brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
    brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)

    champions = []
    for gf_path in genome_files:
        try:
            brain = persist.load_genome(gf_path, brain_input_size, config.BRAIN_HIDDEN_LAYER_SIZE, brain_output_size, rng=rng)
            base = os.path.basename(gf_path); gen_part = base.split('_g')[1].split('_')[0] if '_g' in base else "unk"
            champ_name = f"Gen{gen_part}_{base.split('_fit')[0]}" if '_fit' in base else base.split('.')[0]
            champions.append({'path': gf_path, 'name': champ_name, 'brain': brain, 'wins': 0, 'score': 0.0})
        except Exception as e: print(f"Warning: Could not load {gf_path}: {e}")

    if len(champions) < 2: print("Not enough valid champions loaded."); return
    print(f"Loaded {len(champions)} champions.")

    arena_obj = Arena(config.ARENA_WIDTH, config.ARENA_HEIGHT, wall_bounce_loss_factor=config.WALL_BOUNCE_LOSS_FACTOR, rng=rng)
    match_dt_tourney = config.SIMULATION_DT
    max_steps_tourney = int(config.MATCH_DURATION_SECONDS / match_dt_tourney)

    for count, (c1_idx, c2_idx) in enumerate(itertools.combinations(range(len(champions)), 2)):
        c1, c2 = champions[c1_idx], champions[c2_idx]
        print(f"\nMatch {count+1}: {c1['name']} vs {c2['name']}")
        if visual:
            run_visual_match(c1['path'], c2['path'], num_games=1) # Tournament visual is 1 game
            print(f"Visual match displayed. Score manually if needed for tournament context, or rely on console output.")
            while True:
                score_input = input(f"Score: '1' if {c1['name']} won, '2' if {c2['name']} won, 'd' for draw, 's' to skip scoring: ").strip().lower()
                if score_input == '1': c1['wins']+=1; c1['score']+=1.0; c2['score']-=1.0; break
                elif score_input == '2': c2['wins']+=1; c2['score']+=1.0; c1['score']-=1.0; break
                elif score_input == 'd': c1['score']+=0.1; c2['score']+=0.1; break
                elif score_input == 's': break
                else: print("Invalid input.")
            continue

        # Randomize start positions for tournament matches too
        start_pos_c1 = EvolutionOrchestrator.get_random_start_pose(config.ARENA_WIDTH, config.ARENA_HEIGHT, config.AGENT_RADIUS, rng)
        start_pos_c2 = EvolutionOrchestrator.get_random_start_pose(config.ARENA_WIDTH, config.ARENA_HEIGHT, config.AGENT_RADIUS, rng)

        agent_configs = [
            {'brain': c1['brain'], 'team_id': 1, 'agent_id': c1['name'], 'start_pos': start_pos_c1, 'hp': config.DEFAULT_AGENT_HP},
            {'brain': c2['brain'], 'team_id': 2, 'agent_id': c2['name'], 'start_pos': start_pos_c2, 'hp': config.DEFAULT_AGENT_HP}
        ]
        results = arena_obj.run_match(agent_configs, max_steps_tourney, match_dt_tourney)
        winner = results['winner_team_id']
        if winner == 1: print(f"Winner: {c1['name']}"); c1['wins']+=1; c1['score']+=1.0; c2['score']-=1.0
        elif winner == 2: print(f"Winner: {c2['name']}"); c2['wins']+=1; c2['score']+=1.0; c1['score']-=1.0
        else: print("Draw"); c1['score']+=0.1; c2['score']+=0.1 # Small score for draw

    champions.sort(key=lambda c: (c['score'], c['wins']), reverse=True)
    print("\n--- Tournament Results ---")
    print(f"{'Rank':<5} {'Name':<50} {'Score':<10} {'Wins':<5}")
    print("-" * 70)
    for i, champ in enumerate(champions):
        print(f"{i+1:<5} {champ['name']:<50} {champ['score']:<10.2f} {champ['wins']:<5}")
    if champions: print(f"\nOverall Tournament Winner: {champions[0]['name']} (Score: {champions[0]['score']:.2f}, Wins: {champions[0]['wins']})")


# --- RL Fine-Tuning Functions (largely unchanged from original, but uses config/rand) ---
def run_self_play_episode(genome, arena_width, arena_height, match_max_steps, match_dt, default_hp, rng_ep):
    """Runs a self-play episode for RL fine-tuning."""
    episode_data_for_gradients = []
    self_play_arena = Arena(arena_width, arena_height, rng=rng_ep) # Pass RNG

    brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
    brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)

    # Opponent uses a copy of the genome's weights but its own RNG for decisions if needed by brain internals
    # and its own recurrent state if RNN.
    opponent_brain = TinyNet(
        w_in=genome.w_in.copy(), w_out=genome.w_out.copy(),
        input_size=brain_input_size, hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
        output_size=brain_output_size,
        rng=rng_ep, # Opponent brain gets same master RNG for its construction
        initial_sigma=genome.mutation_sigma # Copy sigma too
    )

    # Randomize start positions for self-play episodes
    start_pos_p1 = EvolutionOrchestrator.get_random_start_pose(arena_width, arena_height, config.AGENT_RADIUS, rng_ep)
    start_pos_p2 = EvolutionOrchestrator.get_random_start_pose(arena_width, arena_height, config.AGENT_RADIUS, rng_ep)


    agent_configs = [
        {'brain': genome, 'team_id': 1, 'agent_id': 'player1_train',
         'start_pos': start_pos_p1, 'hp': default_hp},
        {'brain': opponent_brain, 'team_id': 2, 'agent_id': 'player2_opponent',
         'start_pos': start_pos_p2, 'hp': default_hp}
    ]
    self_play_arena.agents = [] # Clear agents before adding new ones
    player1_agent_body = None

    for i, agent_conf in enumerate(agent_configs):
        start_x, start_y, start_angle = agent_conf.get('start_pos')
        agent = AgentBody(
            x=start_x, y=start_y, angle_deg=start_angle, base_speed=config.AGENT_BASE_SPEED,
            rotation_speed_dps=config.AGENT_ROTATION_SPEED_DPS, radius=config.AGENT_RADIUS,
            agent_id=agent_conf['agent_id'], team_id=agent_conf['team_id'], hp=agent_conf['hp'], brain=agent_conf['brain'],
            weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
            weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
            cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR,
            rng=rng_ep, # Agents get same master RNG
            use_rnn=config.USE_RNN, rnn_hidden_size=config.RNN_HIDDEN_SIZE
        )
        self_play_arena.add_agent(agent)
        if agent_conf['agent_id'] == 'player1_train':
            player1_agent_body = agent

    for step in range(match_max_steps):
        if not player1_agent_body or not player1_agent_body.is_alive(): break

        # For RL, agent needs its own inputs, not from a stale list
        current_inputs_p1 = player1_agent_body.get_inputs_for_nn(self_play_arena.width, self_play_arena.height, self_play_arena.agents)

        x_p1, _h_pre_p1, h_p1_activated, _y_pre_p1, y_p1_actions_plus_recurrent = genome.forward_pass_for_gd(current_inputs_p1)
        episode_data_for_gradients.append({'x_input': x_p1, 'h_activated': h_p1_activated, 'y_activated_actions_plus_recurrent': y_p1_actions_plus_recurrent})

        self_play_arena.update(match_dt) # This calls agent.update which internally calls brain
        match_over, _, _ = self_play_arena.check_match_end_conditions(max_duration_seconds=(match_max_steps * match_dt))
        if match_over: break

    final_results = self_play_arena.check_match_end_conditions(max_duration_seconds=(match_max_steps * match_dt))
    winner_team_id = final_results[1]
    match_reward = 0.0
    if winner_team_id == 1: match_reward = 1.0  # Win
    elif winner_team_id == 2: match_reward = -1.0 # Loss
    # Draw is 0.0

    return episode_data_for_gradients, match_reward


def fine_tune_genome_with_rl(genome_to_tune, num_episodes, learning_rate,
                             arena_width, arena_height,
                             match_max_steps, match_dt,
                             default_hp, rng_ft):
    """Fine-tunes a genome using RL (self-play Reinforce-like)."""
    print(f"\n--- Starting RL Fine-Tuning (Self-Play) for {num_episodes} episodes, LR={learning_rate} ---")
    print(f"Genome initial fitness: {genome_to_tune.fitness:.4f}, initial sigma: {genome_to_tune.mutation_sigma:.4f}")

    avg_rewards_history = []
    for episode_num in range(num_episodes):
        try:
            # Pass the fine-tuning RNG to the episode
            trajectory_data, final_match_reward = run_self_play_episode(
                genome_to_tune, arena_width, arena_height, match_max_steps, match_dt, default_hp, rng_ft
            )
            avg_rewards_history.append(final_match_reward)

            if not trajectory_data:
                print(f"Ep {episode_num+1}/{num_episodes}: No data. Reward: {final_match_reward:.1f}. Skip update.")
                continue

            total_steps_in_episode = len(trajectory_data)

            for step_data in trajectory_data:
                # Policy gradient for actual actions (first NUM_ACTIONS part of output)
                # y_activated_actions = step_data['y_activated_actions_plus_recurrent'][:config.NUM_ACTIONS]
                # The gradient should be w.r.t. all outputs if they all influence reward through policy.
                # However, REINFORCE applies to actions. If recurrent state is just state, not action,
                # then policy gradient should only consider action outputs.
                # For Elman, the recurrent part IS part of the output that becomes input.
                # TinyNet.get_policy_gradient_for_action expects the full y_activated.

                dW_in, dW_out = genome_to_tune.get_policy_gradient(
                    step_data['x_input'],
                    step_data['h_activated'],
                    step_data['y_activated_actions_plus_recurrent'], # Pass the full output vector
                    final_match_reward
                )
                effective_lr = learning_rate / total_steps_in_episode if total_steps_in_episode > 1 else learning_rate
                genome_to_tune.update_weights(dW_in, dW_out, effective_lr)

            if (episode_num + 1) % (max(1, num_episodes // 10)) == 0 or episode_num == 0:
                outcome_str = "WIN" if final_match_reward > 0 else "LOSS" if final_match_reward < 0 else "DRAW"
                avg_rew = np.mean(avg_rewards_history[-(max(1,num_episodes // 20)):]) if avg_rewards_history else 0.0
                print(f"Ep {episode_num+1}/{num_episodes}: Outcome: {outcome_str} ({final_match_reward:.1f}), Steps: {total_steps_in_episode}, AvgRew (last N): {avg_rew:.2f}")
        except Exception as e:
            print(f"Error in RL tuning episode {episode_num+1}: {e}")
            import traceback
            traceback.print_exc()
            break
    print("--- RL Fine-Tuning Finished ---")
    return genome_to_tune


# --- Menu Helper Functions ---
def get_int_input(prompt, default_value):
    while True:
        try: val_str = input(f"{prompt} (default: {default_value}): ").strip(); return default_value if not val_str else int(val_str)
        except ValueError: print("Invalid input. Please enter a whole number.")

def get_float_input(prompt, default_value):
    while True:
        try: val_str = input(f"{prompt} (default: {default_value:.4f}): ").strip(); return default_value if not val_str else float(val_str)
        except ValueError: print("Invalid input. Please enter a number.")

def select_genome_file(prompt_message, allow_none=False, none_option_text="None"):
    print(f"\n{prompt_message}")
    search_dirs = [config.GENOME_STORAGE_DIR, config.BEST_GENOMES_PER_GENERATION_DIR,
                   os.path.join(config.GENOME_STORAGE_DIR, "final_bests"), config.FINETUNED_GENOME_DIR]
    # ... (rest of select_genome_file is largely unchanged, uses config for dirs)
    collected_paths = set()
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for f_name in os.listdir(s_dir):
                if f_name.endswith(".npz"): collected_paths.add(os.path.join(s_dir, f_name))

    sorted_paths = sorted(list(collected_paths), key=lambda p: (os.path.dirname(p), os.path.basename(p)))
    options = []
    if allow_none: options.append((None, none_option_text))
    for path in sorted_paths: options.append((path, f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"))

    if not options and not allow_none and not sorted_paths:
         print(f"No genomes found. Please type a full path to a genome .npz file:")
         while True:
            path_input = input().strip()
            if os.path.exists(path_input) and path_input.endswith(".npz"): return path_input
            else: print("File not found or not a .npz file. Try again:")
    elif (not options and allow_none) or (not sorted_paths and allow_none and len(options) == 1 and options[0][0] is None) :
        print(f"No genomes found in monitored directories.")
        user_path = input(f"Press Enter for '{none_option_text}', or type a full path to a genome: ").strip()
        if not user_path: return None
        if os.path.exists(user_path) and user_path.endswith(".npz"): return user_path
        print("Invalid path specified. Defaulting to 'None'.")
        return None

    for i, (path, display_name) in enumerate(options): print(f"{i}. {display_name}")
    while True:
        raw_choice = input(f"Select by number (0-{len(options)-1}) or type full path: ").strip()
        if os.path.exists(raw_choice) and raw_choice.endswith(".npz"): return raw_choice
        try:
            choice_idx = int(raw_choice)
            if 0 <= choice_idx < len(options): return options[choice_idx][0]
            else: print("Invalid number.")
        except ValueError: print("Invalid input. Please enter a number or a valid file path.")


# --- Menu Mode Functions ---
def menu_run_manual():
    print("\n--- Manual Play Setup ---")
    opponent_genome = select_genome_file("Select AI opponent (optional):", allow_none=True, none_option_text="Random AI/Dummies")
    run_manual_simulation(opponent_genome_path=opponent_genome)

def menu_run_training(current_seed_for_run):
    print("\n--- Configure Training Session ---")
    generations = get_int_input("Number of generations", config.DEFAULT_GENERATIONS)
    pop_size = get_int_input("Population size", config.DEFAULT_POPULATION_SIZE)
    elites = get_int_input("Number of elites", config.DEFAULT_NUM_ELITES)
    mut_sigma = get_float_input("Initial mutation sigma", config.DEFAULT_MUTATION_SIGMA)
    eval_matches = get_int_input("Evaluation matches per genome", config.DEFAULT_EVAL_MATCHES_PER_GENOME)
    sim_dt_chosen = get_float_input("Simulation time step (dt) for training", config.SIMULATION_DT)
    default_match_steps = int(config.MATCH_DURATION_SECONDS / sim_dt_chosen) if sim_dt_chosen > 0 else config.MATCH_MAX_STEPS_TRAINING
    match_steps = get_int_input(f"Max steps per eval match (~{config.MATCH_DURATION_SECONDS}s)", default_match_steps)
    if input("Proceed with training? (y/n): ").strip().lower() == 'y':
        run_training_session(generations, pop_size, elites, mut_sigma, eval_matches, match_steps, sim_dt_chosen, current_seed_for_run)
    else: print("Training cancelled.")

def menu_run_match():
    print("\n--- Visual Match Setup (AI vs AI) ---")
    genome1_path = select_genome_file("Choose Genome 1 (Red):", allow_none=False)
    if not genome1_path: return
    genome2_path = select_genome_file("Choose Genome 2 (Green):", allow_none=False)
    if not genome2_path: return
    if genome1_path == genome2_path: print("Warning: Same genome for both agents.")

    num_games = get_int_input("Number of games to play in this series", 1)
    if num_games < 1: num_games = 1

    run_visual_match(genome1_path, genome2_path, num_games=num_games)

def menu_run_show():
    print("\n--- Show Genome Setup ---")
    genome_path = select_genome_file("Select a genome to showcase:", allow_none=False)
    if not genome_path: return
    run_show_genome(genome_path, scenario='vs_dummies')

def menu_run_post_tournament():
    print("\n--- Post-Training Tournament Setup ---")
    visual = input("Run tournament visually? (y/n, headless is faster for scoring): ").strip().lower() == 'y'
    if visual: print("Visual tournament. You will score each match manually.")
    run_post_training_tournament(visual=visual) # Uses config.BEST_GENOMES_PER_GENERATION_DIR by default

def menu_run_finetune_rl(rng_for_finetune): # Pass RNG specifically for fine-tuning session
    print("\n--- RL Fine-Tuning Setup (Self-Play) ---")
    genome_path = select_genome_file("Select genome to fine-tune:", allow_none=False)
    if not genome_path: return

    brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
    brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)

    try:
        genome_to_fine_tune = persist.load_genome(
            genome_path, brain_input_size, config.BRAIN_HIDDEN_LAYER_SIZE, brain_output_size, rng=rng_for_finetune
        )
        if not isinstance(genome_to_fine_tune, TinyNet):
            print("Error: Loaded object is not a TinyNet instance.")
            return
    except Exception as e:
        print(f"Error loading genome: {e}"); return

    num_episodes_input = get_int_input("Number of self-play episodes for RL tuning", config.DEFAULT_FINETUNE_EPISODES)
    learning_rate_input = get_float_input("Learning rate for RL tuning", config.DEFAULT_FINETUNE_LR)

    base_name, ext_name = os.path.splitext(os.path.basename(genome_path))
    default_save_filename = f"{base_name}_rl_ft{ext_name}"
    if not os.path.exists(config.FINETUNED_GENOME_DIR): os.makedirs(config.FINETUNED_GENOME_DIR)
    default_save_path = os.path.join(config.FINETUNED_GENOME_DIR, default_save_filename)

    save_path_input = input(f"Save fine-tuned genome to (default: {default_save_path}): ").strip()
    if not save_path_input: save_path_input = default_save_path

    print(f"Starting RL fine-tuning for: {genome_path}")
    fine_tuned_genome = fine_tune_genome_with_rl(
        genome_to_tune,
        num_episodes=num_episodes_input,
        learning_rate=learning_rate_input,
        arena_width=config.ARENA_WIDTH, arena_height=config.ARENA_HEIGHT,
        match_max_steps=config.MATCH_MAX_STEPS_TRAINING, match_dt=config.SIMULATION_DT,
        default_hp=config.DEFAULT_AGENT_HP,
        rng_ft=rng_for_finetune # Pass RNG
    )

    save_dir_menu = os.path.dirname(save_path_input)
    save_filename_prefix_menu = os.path.basename(save_path_input).replace(".npz","").split("_fit")[0]
    if not os.path.exists(save_dir_menu): os.makedirs(save_dir_menu)

    # Save with the fine-tuning RNG seed if relevant, or the original training seed.
    # Let's assume we want to log the seed active when this save happens (the main program seed).
    active_seed_for_saving = rand.get_rng()._bit_generator._seed_seq.entropy % (2**32-1) # A bit hacky to get current seed value

    saved_ft_path_menu = persist.save_genome(
        fine_tuned_genome, save_filename_prefix_menu, save_dir_menu,
        fitness=fine_tuned_genome.fitness, # Use updated fitness if RL changed it
        rng_seed_value=active_seed_for_saving
    )
    print(f"RL fine-tuned genome saved to: {saved_ft_path_menu}")
    print("Re-evaluate in arena to see performance changes.")


# --- Main Menu Display and Loop ---
def display_main_menu():
    print("\n===== Evo Arena Main Menu =====")
    print("1. Manual Play vs AI/Dummies")
    print("2. Train New AI Agents (Evolution)")
    print("3. Visual Match (AI vs AI)")
    print("4. Showcase a Trained AI Genome")
    print("5. Run Post-Training Tournament of Champions")
    print("6. Fine-Tune Genome with RL (Self-Play)")
    print("-------------------------------")
    print("0. Exit")
    print("==============================")

def main_menu_loop(current_seed_for_run):
    rng_menu = rand.get_rng() # Get global rng for menu operations if needed
    while True:
        display_main_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1': menu_run_manual()
        elif choice == '2': menu_run_training(current_seed_for_run) # Pass seed
        elif choice == '3': menu_run_match()
        elif choice == '4': menu_run_show()
        elif choice == '5': menu_run_post_tournament()
        elif choice == '6': menu_run_finetune_rl(rng_menu) # Pass RNG for fine-tuning
        elif choice == '0': print("Exiting Evo Arena. Goodbye!"); break
        else: print("Invalid choice, please try again.")

# --- Main Execution ---
def main():
    if not os.path.exists(config.FINETUNED_GENOME_DIR):
        os.makedirs(config.FINETUNED_GENOME_DIR)
    if not os.path.exists(config.BEST_GENOMES_PER_GENERATION_DIR):
        os.makedirs(config.BEST_GENOMES_PER_GENERATION_DIR)
    if not os.path.exists(config.GENOME_STORAGE_DIR):
        os.makedirs(config.GENOME_STORAGE_DIR)


    parser = argparse.ArgumentParser(description="Evo Arena: A simple agent evolution project.")
    parser.add_argument('mode', nargs='?', default=None,
                        choices=['manual', 'train', 'show', 'match', 'tournament', 'finetune_rl'],
                        help="Mode to run. If no mode, menu is shown.")
    # RNG Seed Argument
    parser.add_argument('--seed', type=int, default=None, help="Seed for random number generation.")

    # Evolution args
    parser.add_argument('--generations', type=int, default=config.DEFAULT_GENERATIONS)
    parser.add_argument('--pop_size', type=int, default=config.DEFAULT_POPULATION_SIZE)
    parser.add_argument('--elites', type=int, default=config.DEFAULT_NUM_ELITES)
    parser.add_argument('--mut_sigma', type=float, default=config.DEFAULT_MUTATION_SIGMA)
    parser.add_argument('--eval_matches', type=int, default=config.DEFAULT_EVAL_MATCHES_PER_GENOME)
    parser.add_argument('--sim_dt', type=float, default=config.SIMULATION_DT)
    parser.add_argument('--match_steps', type=int, default=config.MATCH_MAX_STEPS_TRAINING)
    # Manual/Match/Show args
    parser.add_argument('--opponent_genome', type=str, dest='manual_opponent_genome_path')
    parser.add_argument('--g1', type=str, dest='genome1_path', help="Path to genome for AI 1 (Red)")
    parser.add_argument('--g2', type=str, dest='genome2_path', help="Path to genome for AI 2 (Green)")
    parser.add_argument('--num_games', type=int, default=1, help="Number of games for 'match' mode series.")
    parser.add_argument('--genome', type=str, dest='show_genome_path')
    parser.add_argument('--scenario', type=str, default='vs_dummies', choices=['vs_dummies'])
    # Tournament args
    parser.add_argument('--tournament_dir', type=str, default=config.BEST_GENOMES_PER_GENERATION_DIR)
    parser.add_argument('--visual_tournament', action='store_true')
    # Fine-tuning args (RL)
    parser.add_argument('--genome_path_for_finetune', type=str, help="Path to .npz genome for fine-tuning.")
    parser.add_argument('--finetune_episodes', type=int, default=config.DEFAULT_FINETUNE_EPISODES)
    parser.add_argument('--finetune_lr', type=float, default=config.DEFAULT_FINETUNE_LR)
    parser.add_argument('--save_finetuned_path', type=str, help="Path to save fine-tuned genome (e.g., storage/genomes/finetuned/my_ft_genome.npz).")

    args = parser.parse_args()

    # Initialize RNG once, early using rand.py
    # This will print the seed if it's newly generated.
    current_seed = rand.set_seed(args.seed)
    main_rng = rand.get_rng() # Get the initialized global RNG

    if args.mode is None:
        main_menu_loop(current_seed) # Pass the seed for logging in training if started from menu
        return

    # Adjust match_steps for training if sim_dt is custom
    current_match_steps_for_training = args.match_steps
    if args.mode == 'train':
        sim_dt_is_custom = (args.sim_dt != config.SIMULATION_DT)
        # Check if match_steps was the default calculated one based on the *original* config.SIMULATION_DT
        match_steps_is_default_calc_for_training = (args.match_steps == int(config.MATCH_DURATION_SECONDS / config.SIMULATION_DT))

        if sim_dt_is_custom and match_steps_is_default_calc_for_training:
            if args.sim_dt > 0:
                current_match_steps_for_training = int(config.MATCH_DURATION_SECONDS / args.sim_dt)
                print(f"Note: --sim_dt changed for training. Adjusting --match_steps from {args.match_steps} to {current_match_steps_for_training} to maintain ~{config.MATCH_DURATION_SECONDS}s match duration.")
            else:
                print(f"Warning: Invalid --sim_dt ({args.sim_dt}) for training. Using default match_steps ({current_match_steps_for_training}).")

    if args.mode == 'manual':
        run_manual_simulation(opponent_genome_path=args.manual_opponent_genome_path)
    elif args.mode == 'train':
        run_training_session(args.generations, args.pop_size, args.elites, args.mut_sigma, args.eval_matches, current_match_steps_for_training, args.sim_dt, current_seed)
    elif args.mode == 'match':
        if not (args.genome1_path and args.genome2_path): parser.error("'match' mode requires --g1 and --g2.")
        if not os.path.exists(args.genome1_path): parser.error(f"Genome file not found for --g1: {args.genome1_path}")
        if not os.path.exists(args.genome2_path): parser.error(f"Genome file not found for --g2: {args.genome2_path}")
        num_games_cli = args.num_games if args.num_games >=1 else 1
        run_visual_match(args.genome1_path, args.genome2_path, num_games=num_games_cli)
    elif args.mode == 'show':
        if not args.show_genome_path: parser.error("'show' mode requires --genome.")
        if not os.path.exists(args.show_genome_path): parser.error(f"Genome file not found for --genome: {args.show_genome_path}")
        run_show_genome(args.show_genome_path, scenario=args.scenario)
    elif args.mode == 'tournament':
        run_post_training_tournament(tournament_genome_dir=args.tournament_dir, visual=args.visual_tournament)
    elif args.mode == 'finetune_rl':
        if not args.genome_path_for_finetune: parser.error("'finetune_rl' mode requires --genome_path_for_finetune.")
        if not os.path.exists(args.genome_path_for_finetune): parser.error(f"Genome file not found for fine-tuning: {args.genome_path_for_finetune}")

        brain_input_size = config.BRAIN_SENSORY_INPUTS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
        brain_output_size = config.NUM_ACTIONS + (config.RNN_HIDDEN_SIZE if config.USE_RNN else 0)
        try:
            genome_to_ft = persist.load_genome(
                args.genome_path_for_finetune, brain_input_size, config.BRAIN_HIDDEN_LAYER_SIZE, brain_output_size, rng=main_rng
            )
            if not isinstance(genome_to_ft, TinyNet): parser.error("Loaded object is not a TinyNet instance.")
        except Exception as e: parser.error(f"Error loading genome: {e}")

        print(f"[DEBUG CLI] Calling fine_tune_genome_with_rl with episodes: {args.finetune_episodes}, lr: {args.finetune_lr}")
        fine_tuned_g = fine_tune_genome_with_rl(
            genome_to_ft,
            num_episodes=args.finetune_episodes,
            learning_rate=args.finetune_lr,
            arena_width=config.ARENA_WIDTH, arena_height=config.ARENA_HEIGHT,
            match_max_steps=config.MATCH_MAX_STEPS_TRAINING, match_dt=config.SIMULATION_DT,
            default_hp=config.DEFAULT_AGENT_HP,
            rng_ft=main_rng # Pass the main RNG
        )

        save_path_ft = args.save_finetuned_path
        if not save_path_ft:
            base, ext = os.path.splitext(args.genome_path_for_finetune)
            save_path_ft = os.path.join(config.FINETUNED_GENOME_DIR, f"{os.path.basename(base)}_rl_ft{ext}")

        ft_save_dir = os.path.dirname(save_path_ft)
        ft_filename_prefix = os.path.basename(save_path_ft).replace(".npz","").split("_fit")[0]
        if not os.path.exists(ft_save_dir): os.makedirs(ft_save_dir)

        saved_ft_final_path = persist.save_genome(
            fine_tuned_g, ft_filename_prefix, ft_save_dir,
            fitness=fine_tuned_g.fitness, rng_seed_value=current_seed
        )
        print(f"RL fine-tuned genome saved to: {saved_ft_final_path}")
        print("Re-evaluate performance in the arena.")
    else: # If sys.argv > 1 but mode is not recognized (should be caught by parser choices)
        main_menu_loop(current_seed)


if __name__ == '__main__':
    main()