# rand.py
import numpy as np
import config # Assuming config.py is at the root

global_rng = None
_seed_used_for_global_rng = None # Store the seed for logging/metadata

def set_seed(seed_value=None):
    """
    Initializes or re-initializes the global RNG with a specific seed.
    If seed_value is None, it uses config.DEFAULT_RNG_SEED.
    If config.DEFAULT_RNG_SEED is also None, a random seed is generated.
    The actual seed used is stored and returned.
    """
    global global_rng, _seed_used_for_global_rng

    if seed_value is not None:
        final_seed = int(seed_value)
    elif config.DEFAULT_RNG_SEED is not None:
        final_seed = int(config.DEFAULT_RNG_SEED)
    else:
        # Generate a random seed if none is provided or configured
        # Use SeedSequence for good quality random seeds
        sq = np.random.SeedSequence()
        final_seed = sq.entropy % (2**32 -1) # Keep it within typical int32 range for np.random.default_rng
        print(f"INFO: No seed provided via CLI or config.DEFAULT_RNG_SEED. Generated random seed: {final_seed}")

    global_rng = np.random.default_rng(final_seed)
    _seed_used_for_global_rng = final_seed
    # print(f"Global RNG seeded with: {final_seed}") # Optional: for debugging
    return final_seed

def get_rng():
    """Returns the globally managed RNG instance. Initializes if not already done."""
    global global_rng
    if global_rng is None:
        # This case should ideally be avoided by calling set_seed() early in main.
        print("Warning: Global RNG accessed before explicit seeding. Seeding with default rules now.")
        set_seed() # This will use config.DEFAULT_RNG_SEED or generate a new one
    return global_rng

def get_current_seed():
    """Returns the seed that was used to initialize the global RNG."""
    return _seed_used_for_global_rng