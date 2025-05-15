# evo_arena/storage/persist.py
import numpy as np
import os
import json # For match replays later (not implemented yet fully)

from agents.brain import TinyNet # To reconstruct TinyNet objects
import config # For default brain params if needed by TinyNet constructor

def save_genome(genome_brain, filename_prefix="genome", directory="storage/genomes",
                generation=None, fitness=None, rng_seed_value=None): # Added rng_seed_value
    """
    Saves the genome (weights and sigma of a TinyNet) to a .npz file.
    Includes RNG seed in metadata.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    w_in, w_out, mutation_sigma = genome_brain.get_genome_params() # Now also gets sigma

    # Construct filename
    filename = f"{filename_prefix}"
    if generation is not None:
        filename += f"_g{generation:05d}"
    if fitness is not None:
        filename += f"_fit{fitness:.3f}"
    filename += ".npz"

    filepath = os.path.join(directory, filename)

    save_data = {
        'w_in': w_in,
        'w_out': w_out,
        'mutation_sigma': mutation_sigma # Save self-adaptive sigma
    }
    # Use genome_brain.fitness if fitness arg not provided, otherwise use arg
    current_fitness = fitness if fitness is not None else genome_brain.fitness
    save_data['fitness'] = current_fitness

    if rng_seed_value is not None:
        save_data['rng_seed'] = np.array(rng_seed_value) # Store as numpy array for savez

    np.savez(filepath, **save_data)
    # print(f"Saved genome to {filepath}") # Less verbose during training
    return filepath

def load_genome(filepath, input_size=None, hidden_size=None, output_size=None, rng=None):
    """
    Loads a genome from a .npz file and reconstructs a TinyNet.
    Uses config defaults for sizes if not provided.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Genome file not found: {filepath}")

    data = np.load(filepath)
    w_in = data['w_in']
    w_out = data['w_out']

    # NN Structure: try to infer from weights if not provided, or use config defaults
    # This assumes saved weights correspond to the current config if sizes are None.
    _input_size = input_size if input_size is not None else config.BRAIN_INPUT_SIZE
    _hidden_size = hidden_size if hidden_size is not None else config.BRAIN_HIDDEN_LAYER_SIZE
    _output_size = output_size if output_size is not None else config.BRAIN_OUTPUT_SIZE

    # Load self-adaptive sigma if present, else use default from config
    initial_sigma_loaded = float(data['mutation_sigma']) if 'mutation_sigma' in data else config.DEFAULT_MUTATION_SIGMA

    loaded_brain = TinyNet(w_in, w_out,
                           input_size=_input_size,
                           hidden_size=_hidden_size,
                           output_size=_output_size,
                           rng=rng, # Pass RNG for consistency if TinyNet uses it internally on init
                           initial_sigma=initial_sigma_loaded)

    if 'fitness' in data:
        loaded_brain.fitness = float(data['fitness'])
    
    if 'rng_seed' in data:
        # print(f"Genome was saved with RNG seed: {data['rng_seed']}") # Optional info
        pass # Seed is just metadata here, not used to re-seed global RNG on load

    # print(f"Loaded genome from {filepath}") # Less verbose
    return loaded_brain

# --- Match Replay Functions (for later, as per spec section 6) ---
REPLAY_DIR = "storage/replays"

def start_match_replay(filename_prefix="match_replay"):
    if not os.path.exists(REPLAY_DIR):
        os.makedirs(REPLAY_DIR)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{filename_prefix}_{timestamp}.jsonl"
    filepath = os.path.join(REPLAY_DIR, filename)
    return open(filepath, 'w'), filepath # Return file handle and path

def record_arena_snapshot(replay_file_handle, game_time, agents_data):
    """
    Records a snapshot of the arena state to the replay file.
    agents_data: A list of dicts, each representing an agent's state.
    """
    snapshot = {
        't': round(game_time, 3),
        'state': agents_data 
    }
    replay_file_handle.write(json.dumps(snapshot) + '\n')

def close_match_replay(replay_file_handle):
    if replay_file_handle:
        replay_file_handle.close()

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Test saving and loading
    test_brain = TinyNet()
    test_brain.fitness = 123.456
    
    # Create dummy directories if they don't exist
    if not os.path.exists("storage/genomes"):
        os.makedirs("storage/genomes")

    filepath = save_genome(test_brain, filename_prefix="test_dummy", generation=0, fitness=test_brain.fitness)
    loaded_brain = load_genome(filepath)
    
    assert np.array_equal(test_brain.w_in, loaded_brain.w_in)
    assert np.array_equal(test_brain.w_out, loaded_brain.w_out)
    assert hasattr(loaded_brain, 'fitness') and loaded_brain.fitness == test_brain.fitness
    print("Save/Load test successful.")

    # Test replay (rudimentary)
    # file_handle, replay_path = start_match_replay()
    # print(f"Replay file started: {replay_path}")
    # record_arena_snapshot(file_handle, 0.02, [{'id': 'a1', 'x': 10, 'y': 20, 'hp': 100}])
    # record_arena_snapshot(file_handle, 0.04, [{'id': 'a1', 'x': 12, 'y': 22, 'hp': 90}])
    # close_match_replay(file_handle)
    # print("Replay test successful.")