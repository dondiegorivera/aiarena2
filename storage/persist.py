# evo_arena/storage/persist.py
import numpy as np
import os
import json 

from agents.brain import TinyNet 
import config 

def save_genome(genome_brain, filename_prefix="genome", directory="storage/genomes",
                generation=None, fitness=None, rng_seed_value=None): 
    """
    Saves the genome (weights and sigma of a TinyNet) to a .npz file.
    Includes RNG seed in metadata.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    w_in, w_out, mutation_sigma = genome_brain.get_genome_params() 

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
        'mutation_sigma': mutation_sigma 
    }
    current_fitness = fitness if fitness is not None else genome_brain.fitness
    save_data['fitness'] = current_fitness

    if rng_seed_value is not None:
        save_data['rng_seed'] = np.array(rng_seed_value) 

    np.savez(filepath, **save_data)
    return filepath

def load_genome(filepath, input_size=None, hidden_size=None, output_size=None, rng=None):
    """
    Loads a genome from a .npz file and reconstructs a TinyNet.
    If input_size/output_size are None when calling this, TinyNet constructor will attempt to infer from weights.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Genome file not found: {filepath}")

    data = np.load(filepath)
    w_in_loaded = data['w_in']
    w_out_loaded = data['w_out']

    # Pass None for input_size and output_size to TinyNet constructor.
    # It will infer these from the shapes of w_in_loaded and w_out_loaded.
    # hidden_size is typically a global config or could be saved in the genome if it were variable.
    _hidden_size = hidden_size if hidden_size is not None else config.BRAIN_HIDDEN_LAYER_SIZE
    
    initial_sigma_loaded = float(data['mutation_sigma']) if 'mutation_sigma' in data else config.DEFAULT_MUTATION_SIGMA

    # When loading, TinyNet will use w_in_loaded.shape[1] for its input_size,
    # and w_out_loaded.shape[0] for its output_size.
    loaded_brain = TinyNet(w_in=w_in_loaded, w_out=w_out_loaded,
                           input_size=None,     # Signal TinyNet to infer from w_in_loaded
                           hidden_size=_hidden_size,
                           output_size=None,    # Signal TinyNet to infer from w_out_loaded
                           rng=rng, 
                           initial_sigma=initial_sigma_loaded)

    if 'fitness' in data:
        loaded_brain.fitness = float(data['fitness'])
    
    if 'rng_seed' in data:
        # print(f"DEBUG: Genome {os.path.basename(filepath)} was saved with RNG seed: {data['rng_seed']}")
        pass 

    # Debug: Check if inferred sizes match expectations from current config (if they were passed for comparison)
    # This is useful for verifying consistency if you also pass current config sizes.
    # if input_size is not None and loaded_brain.input_size != input_size:
    #     print(f"DEBUG Load Genome: Loaded brain input_size {loaded_brain.input_size} (from weights) "
    #           f"differs from provided/current config input_size {input_size}.")
    # if output_size is not None and loaded_brain.output_size != output_size:
    #     print(f"DEBUG Load Genome: Loaded brain output_size {loaded_brain.output_size} (from weights) "
    #           f"differs from provided/current config output_size {output_size}.")


    return loaded_brain


REPLAY_DIR = "storage/replays"

def start_match_replay(filename_prefix="match_replay"):
    if not os.path.exists(REPLAY_DIR):
        os.makedirs(REPLAY_DIR)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{filename_prefix}_{timestamp}.jsonl"
    filepath = os.path.join(REPLAY_DIR, filename)
    return open(filepath, 'w'), filepath

def record_arena_snapshot(replay_file_handle, game_time, agents_data):
    snapshot = {
        't': round(game_time, 3),
        'state': agents_data 
    }
    replay_file_handle.write(json.dumps(snapshot) + '\n')

def close_match_replay(replay_file_handle):
    if replay_file_handle:
        replay_file_handle.close()

if __name__ == '__main__':
    if not os.path.exists("storage/genomes"):
        os.makedirs("storage/genomes")

    # Test: Create a dummy RNG for testing TinyNet/load/save
    test_rng = np.random.default_rng(123)

    # Test with an RNN-like structure
    print("Testing RNN-like structure save/load:")
    sensory_inputs = 15
    rnn_hidden = 16
    num_actions = 4
    test_input_rnn = sensory_inputs + rnn_hidden
    test_output_rnn = num_actions + rnn_hidden
    
    test_brain_rnn = TinyNet(input_size=test_input_rnn, hidden_size=10, output_size=test_output_rnn, rng=test_rng)
    test_brain_rnn.fitness = 123.456
    filepath_rnn = save_genome(test_brain_rnn, filename_prefix="test_dummy_rnn", generation=0, fitness=test_brain_rnn.fitness, rng_seed_value=123)
    
    # When loading, we might not know the exact original sizes if they weren't default config
    # So, load_genome passes None for sizes, letting TinyNet infer from weights.
    loaded_brain_rnn = load_genome(filepath_rnn, rng=test_rng) 
    
    assert np.array_equal(test_brain_rnn.w_in, loaded_brain_rnn.w_in)
    assert np.array_equal(test_brain_rnn.w_out, loaded_brain_rnn.w_out)
    assert loaded_brain_rnn.input_size == test_input_rnn, f"RNN Load Error: Input size mismatch. Expected {test_input_rnn}, got {loaded_brain_rnn.input_size}"
    assert loaded_brain_rnn.output_size == test_output_rnn, f"RNN Load Error: Output size mismatch. Expected {test_output_rnn}, got {loaded_brain_rnn.output_size}"
    assert hasattr(loaded_brain_rnn, 'fitness') and loaded_brain_rnn.fitness == test_brain_rnn.fitness
    print("RNN-like Save/Load test successful.")

    # Test with an MLP-like structure (assuming config.USE_RNN could be false)
    print("\nTesting MLP-like structure save/load:")
    test_input_mlp = sensory_inputs
    test_output_mlp = num_actions
    test_brain_mlp = TinyNet(input_size=test_input_mlp, hidden_size=10, output_size=test_output_mlp, rng=test_rng)
    test_brain_mlp.fitness = 789.012
    filepath_mlp = save_genome(test_brain_mlp, filename_prefix="test_dummy_mlp", generation=1, fitness=test_brain_mlp.fitness, rng_seed_value=456)
    loaded_brain_mlp = load_genome(filepath_mlp, rng=test_rng)

    assert np.array_equal(test_brain_mlp.w_in, loaded_brain_mlp.w_in)
    assert np.array_equal(test_brain_mlp.w_out, loaded_brain_mlp.w_out)
    assert loaded_brain_mlp.input_size == test_input_mlp, f"MLP Load Error: Input size mismatch. Expected {test_input_mlp}, got {loaded_brain_mlp.input_size}"
    assert loaded_brain_mlp.output_size == test_output_mlp, f"MLP Load Error: Output size mismatch. Expected {test_output_mlp}, got {loaded_brain_mlp.output_size}"
    assert hasattr(loaded_brain_mlp, 'fitness') and loaded_brain_mlp.fitness == test_brain_mlp.fitness
    print("MLP-like Save/Load test successful.")