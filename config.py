# config.py
import numpy as np

# --- RNG Configuration ---
DEFAULT_RNG_SEED = None # Set to an integer for reproducible runs, None for random seed

# --- Arena Configuration ---
ARENA_WIDTH = 800.0
ARENA_HEIGHT = 800.0
WALL_BOUNCE_LOSS_FACTOR = 0.9 # Percentage of speed kept after bouncing

# --- Simulation Configuration ---
SIMULATION_DT = 1.0 / 30.0  # Time step for headless simulation (training)
VISUAL_FPS = 50             # Target FPS for visual modes
MATCH_DURATION_SECONDS = 60.0

# --- Agent Configuration ---
AGENT_RADIUS = 15.0
AGENT_BASE_SPEED = 150.0
AGENT_ROTATION_SPEED_DPS = 180.0 # Degrees per second
DEFAULT_AGENT_HP = 100.0
COOLDOWN_JITTER_FACTOR = 0.1 # e.g., 0.1 for +/-10% weapon cooldown variation

# --- Agent Perception & Control (EXPANDED_PLAN §3) ---
LIDAR_NUM_RAYS = 12
LIDAR_MAX_RANGE = 400.0 # Max distance for lidar rays
# Input vector: [LIDAR_NUM_RAYS distances, own_health, weapon_ready, bias_input]
SENSORY_INPUT_SIZE = LIDAR_NUM_RAYS + 3 # Health, Weapon Ready, Bias
NUM_ACTIONS = 4 # Thrust, Strafe, Rotate, Fire
FIRE_THRESHOLD = 0.5 # For continuous output mapped to trigger probability
STRAFE_SPEED_FACTOR = 0.75 # Agent strafes at this factor of base_speed

# --- RNN Configuration (EXPANDED_PLAN §3) ---
USE_RNN = True # Toggle for using RNN architecture
RNN_HIDDEN_SIZE = 16 # Size of the recurrent hidden state vector

# --- Weapon Configuration ---
WEAPON_RANGE = 150.0
WEAPON_ARC_DEG = 90.0
WEAPON_COOLDOWN_TIME = 0.6 # Base cooldown time in seconds
WEAPON_DAMAGE = 25.0

# --- Evolution Configuration ---
DEFAULT_GENERATIONS = 20
DEFAULT_POPULATION_SIZE = 32
DEFAULT_NUM_ELITES = 4
DEFAULT_MUTATION_SIGMA = 0.2 # Initial mutation sigma for weights
DEFAULT_EVAL_MATCHES_PER_GENOME = 4 # Number of evaluation matches per genome
MATCH_MAX_STEPS_TRAINING = int(MATCH_DURATION_SECONDS / SIMULATION_DT)

# Fitness Signal (EXPANDED_PLAN §2)
C_DAMAGE = 0.01     # Coefficient for damage dealt in fitness
C_SURVIVAL = 0.001  # Coefficient for ticks survived in fitness

# Evolution Upgrades (EXPANDED_PLAN §4)
SIGMA_ADAPTATION_TAU = 1.0 / np.sqrt(2 * np.sqrt(300)) # Heuristic for sigma adaptation (300 is approx num_weights)
TOURNAMENT_SIZE_K = 5 # Number of individuals in a tournament selection round

# --- Storage Configuration ---
GENOME_STORAGE_DIR = "storage/genomes"
BEST_GENOMES_PER_GENERATION_DIR = "storage/genomes/per_generation_bests"
FINETUNED_GENOME_DIR = "storage/genomes/finetuned" # From existing main.py

# --- UI & Display Configuration ---
MANUAL_AGENT_COLOR = (0, 150, 255)
PLAYER_TEAM_ID = 1
DUMMY_AGENT_COLOR = (255, 100, 0)
DUMMY_TEAM_ID = 2
AI_OPPONENT_COLOR = (220, 0, 0)
AI_OPPONENT_TEAM_ID = 3
AI_AGENT_COLOR_2 = (0, 200, 50)

# --- TinyNet Brain Configuration ---
# Determine input and output sizes based on features
BRAIN_SENSORY_INPUTS = LIDAR_NUM_RAYS + 3 # + 1 (_health) + 1 (_weapon_ready) + 1 (_bias)

if USE_RNN:
    BRAIN_INPUT_SIZE = BRAIN_SENSORY_INPUTS + RNN_HIDDEN_SIZE
    BRAIN_OUTPUT_SIZE = NUM_ACTIONS + RNN_HIDDEN_SIZE
else:
    BRAIN_INPUT_SIZE = BRAIN_SENSORY_INPUTS
    BRAIN_OUTPUT_SIZE = NUM_ACTIONS

BRAIN_HIDDEN_LAYER_SIZE = 16 # As per original TinyNet, can be configured

# For RL Fine-tuning (from existing main.py, can be configured via CLI too)
DEFAULT_FINETUNE_EPISODES = 50
DEFAULT_FINETUNE_LR = 0.001