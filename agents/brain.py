# evo_arena/agents/brain.py
import numpy as np
import config # Import config for defaults

class TinyNet:
    def __init__(self, w_in=None, w_out=None,
                 input_size=None, # Allow None to infer from w_in
                 hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
                 output_size=None, # Allow None to infer from w_out
                 rng=None, # Expect an RNG instance
                 initial_sigma=config.DEFAULT_MUTATION_SIGMA): # For self-adaptive sigma

        self.rng = rng if rng is not None else np.random.default_rng()
        self.hidden_size = hidden_size # Hidden size is usually fixed or part of genome

        # Determine actual input_size
        if w_in is not None:
            self.w_in = np.array(w_in, dtype=np.float64)
            actual_input_size = self.w_in.shape[1]
            if input_size is not None and input_size != actual_input_size:
                print(f"Warning: TinyNet constructor input_size ({input_size}) "
                      f"differs from w_in.shape[1] ({actual_input_size}). Using w_in's dimension.")
            self.input_size = actual_input_size
        else:
            # Use provided input_size or default from config if creating new network
            self.input_size = input_size if input_size is not None else config.BRAIN_INPUT_SIZE
            self.w_in = self.rng.uniform(-1, 1, (self.hidden_size, self.input_size)).astype(np.float64)

        # Determine actual output_size
        if w_out is not None:
            self.w_out = np.array(w_out, dtype=np.float64)
            actual_output_size = self.w_out.shape[0]
            if output_size is not None and output_size != actual_output_size:
                 print(f"Warning: TinyNet constructor output_size ({output_size}) "
                       f"differs from w_out.shape[0] ({actual_output_size}). Using w_out's dimension.")
            self.output_size = actual_output_size
        else:
            # Use provided output_size or default from config if creating new network
            self.output_size = output_size if output_size is not None else config.BRAIN_OUTPUT_SIZE
            self.w_out = self.rng.uniform(-1, 1, (self.output_size, self.hidden_size)).astype(np.float64)

        self.fitness = 0.0
        self.mutation_sigma = float(initial_sigma) # For self-adaptive mutation strength

    def __call__(self, x_full_input):
        """
        Standard forward pass for evaluation/evolution.
        x_full_input already includes sensory inputs and recurrent state if applicable.
        """
        if not isinstance(x_full_input, np.ndarray):
            x_full_input = np.array(x_full_input, dtype=np.float64)

        if x_full_input.shape[0] != self.input_size:
            # This is a critical error. The AgentBody should provide input matching self.input_size.
            # self.input_size is now correctly set from w_in.shape[1] if w_in was provided.
            error_msg = (f"CRITICAL ERROR: TinyNet input size mismatch in __call__. "
                         f"Network's self.input_size is {self.input_size} (derived from w_in shape: {self.w_in.shape if hasattr(self, 'w_in') else 'N/A'}), "
                         f"but received input x_full_input with shape {x_full_input.shape}. "
                         "This indicates AgentBody is not correctly inferring or using the brain's expected input structure. "
                         "Ensure AgentBody correctly sets its 'use_rnn' flag based on the loaded brain's properties.")
            raise ValueError(error_msg)

        h_pre_activation = self.w_in @ x_full_input
        h_activated = np.tanh(h_pre_activation)

        y_pre_activation = self.w_out @ h_activated
        y_full_activated = np.tanh(y_pre_activation)

        return y_full_activated


    def forward_pass_for_gd(self, x_full_input):
        """Forward pass that returns intermediate activations needed for GD/RL."""
        if not isinstance(x_full_input, np.ndarray):
            x_full_input = np.array(x_full_input, dtype=np.float64)

        if x_full_input.shape[0] != self.input_size:
            error_msg = (f"CRITICAL ERROR: TinyNet input size mismatch in forward_pass_for_gd. "
                         f"Network's self.input_size is {self.input_size}, "
                         f"but received input x_full_input with shape {x_full_input.shape}.")
            raise ValueError(error_msg)

        h_pre_activation = self.w_in @ x_full_input
        h_activated = np.tanh(h_pre_activation)

        y_pre_activation = self.w_out @ h_activated
        y_full_activated = np.tanh(y_pre_activation) 

        return x_full_input, h_pre_activation, h_activated, y_pre_activation, y_full_activated


    def get_policy_gradient(self, x_input, h_activated, y_full_activated, match_reward):
        """
        Calculates a HEURISTIC pseudo-gradient for a REINFORCE-like update.
        Applies to the full output vector (actions + recurrent part if RNN).
        """
        error_signal_on_output_activations = match_reward * y_full_activated
        delta_output_layer_pre_activation = error_signal_on_output_activations * (1 - y_full_activated**2)
        dW_out = np.outer(delta_output_layer_pre_activation, h_activated)
        error_hidden_layer_activation = self.w_out.T @ delta_output_layer_pre_activation
        delta_hidden_layer_pre_activation = error_hidden_layer_activation * (1 - h_activated**2)
        dW_in = np.outer(delta_hidden_layer_pre_activation, x_input)
        return dW_in, dW_out

    def update_weights(self, dW_in, dW_out, learning_rate):
        """Updates weights using the calculated gradients."""
        self.w_in -= learning_rate * dW_in
        self.w_out -= learning_rate * dW_out

    def mutate(self, mutation_rate_weights=1.0): # Sigma is now internal
        """Mutates the network's sigma and then its weights. Returns a new mutated TinyNet."""
        new_sigma = self.mutation_sigma * np.exp(config.SIGMA_ADAPTATION_TAU * self.rng.normal(0, 1))
        new_sigma = max(1e-5, new_sigma) 

        w_in_mutated = self.w_in.copy()
        w_out_mutated = self.w_out.copy()

        if self.rng.random() < mutation_rate_weights: 
            noise_in = self.rng.normal(0, new_sigma, self.w_in.shape).astype(np.float64)
            w_in_mutated += noise_in

        if self.rng.random() < mutation_rate_weights: 
            noise_out = self.rng.normal(0, new_sigma, self.w_out.shape).astype(np.float64)
            w_out_mutated += noise_out
        
        mutated_net = TinyNet(w_in=w_in_mutated, w_out=w_out_mutated, # Pass mutated weights
                              input_size=self.input_size, # Child inherits parent's structure
                              hidden_size=self.hidden_size,
                              output_size=self.output_size,
                              rng=self.rng, initial_sigma=new_sigma) 
        mutated_net.fitness = 0.0 
        return mutated_net


    @classmethod
    def crossover(cls, parent1, parent2, rng_instance=None):
        """Performs uniform crossover between two parent TinyNets."""
        if rng_instance is None: 
            rng_instance = np.random.default_rng()

        if not (parent1.input_size == parent2.input_size and \
                parent1.hidden_size == parent2.hidden_size and \
                parent1.output_size == parent2.output_size and \
                parent1.w_in.shape == parent2.w_in.shape and \
                parent1.w_out.shape == parent2.w_out.shape):
            print("Critical Warning: Crossover between parents with different NN structures attempted. "
                  f"P1(I:{parent1.input_size}, H:{parent1.hidden_size}, O:{parent1.output_size}, "
                  f"Win:{parent1.w_in.shape}, Wout:{parent1.w_out.shape}) vs "
                  f"P2(I:{parent2.input_size}, H:{parent2.hidden_size}, O:{parent2.output_size}, "
                  f"Win:{parent2.w_in.shape}, Wout:{parent2.w_out.shape}). Returning copy of parent1.")
            return cls(w_in=parent1.w_in.copy(), w_out=parent1.w_out.copy(),
                       input_size=parent1.input_size, hidden_size=parent1.hidden_size, 
                       output_size=parent1.output_size,
                       rng=rng_instance, initial_sigma=parent1.mutation_sigma)


        mask_in = rng_instance.random(parent1.w_in.shape) < 0.5
        w_in_child = np.where(mask_in, parent1.w_in, parent2.w_in)

        mask_out = rng_instance.random(parent1.w_out.shape) < 0.5
        w_out_child = np.where(mask_out, parent1.w_out, parent2.w_out)

        child_sigma = parent1.mutation_sigma 

        child_net = cls(w_in=w_in_child, w_out=w_out_child, # Pass child's weights
                        input_size=parent1.input_size, # Child has same structure as parents
                        hidden_size=parent1.hidden_size,
                        output_size=parent1.output_size,
                        rng=rng_instance, initial_sigma=child_sigma)
        child_net.fitness = 0.0 
        return child_net


    def get_genome_params(self):
        """Returns parameters that define the genome for saving/loading."""
        return self.w_in, self.w_out, self.mutation_sigma