# evo_arena/agents/brain.py
import numpy as np
import config # Import config for defaults

class TinyNet:
    def __init__(self, w_in=None, w_out=None,
                 input_size=config.BRAIN_INPUT_SIZE,
                 hidden_size=config.BRAIN_HIDDEN_LAYER_SIZE,
                 output_size=config.BRAIN_OUTPUT_SIZE,
                 rng=None, # Expect an RNG instance
                 initial_sigma=config.DEFAULT_MUTATION_SIGMA): # For self-adaptive sigma

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rng = rng if rng is not None else np.random.default_rng() # Fallback, but should be passed

        if w_in is not None:
            self.w_in = np.array(w_in, dtype=np.float64)
        else:
            self.w_in = self.rng.uniform(-1, 1, (self.hidden_size, self.input_size)).astype(np.float64)

        if w_out is not None:
            self.w_out = np.array(w_out, dtype=np.float64)
        else:
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
            # This indicates a potential problem in how AgentBody prepares inputs
            # For robustness, pad or truncate, but ideally this shouldn't be hit.
            # print(f"Warning: TinyNet input size mismatch. Expected {self.input_size}, got {x_full_input.shape[0]}. Adjusting.")
            if x_full_input.shape[0] < self.input_size:
                x_padded = np.zeros(self.input_size, dtype=np.float64)
                x_padded[:x_full_input.shape[0]] = x_full_input
                x_full_input = x_padded
            else: # x.shape[0] > self.input_size
                x_full_input = x_full_input[:self.input_size]

        h_pre_activation = self.w_in @ x_full_input
        h_activated = np.tanh(h_pre_activation)

        y_pre_activation = self.w_out @ h_activated
        # For continuous control, actions are tanh. Recurrent state usually also tanh.
        y_full_activated = np.tanh(y_pre_activation)

        return y_full_activated


    def forward_pass_for_gd(self, x_full_input):
        """Forward pass that returns intermediate activations needed for GD/RL."""
        if not isinstance(x_full_input, np.ndarray):
            x_full_input = np.array(x_full_input, dtype=np.float64)

        if x_full_input.shape[0] != self.input_size: # Same robustness as __call__
            if x_full_input.shape[0] < self.input_size:
                x_padded = np.zeros(self.input_size, dtype=np.float64)
                x_padded[:x_full_input.shape[0]] = x_full_input
                x_full_input = x_padded
            else:
                x_full_input = x_full_input[:self.input_size]

        h_pre_activation = self.w_in @ x_full_input
        h_activated = np.tanh(h_pre_activation)

        y_pre_activation = self.w_out @ h_activated
        y_full_activated = np.tanh(y_pre_activation) # Apply tanh to all outputs

        return x_full_input, h_pre_activation, h_activated, y_pre_activation, y_full_activated


    def get_policy_gradient(self, x_input, h_activated, y_full_activated, match_reward):
        """
        Calculates a HEURISTIC pseudo-gradient for a REINFORCE-like update.
        Applies to the full output vector (actions + recurrent part if RNN).
        """
        # Error signal for y_full_activated (all outputs). Push towards its sign if reward is positive.
        # This is a very basic REINFORCE style update.
        # A more standard REINFORCE for continuous actions often involves a Gaussian policy,
        # but here we have deterministic tanh outputs.
        # Grad log pi (a|s) * R. For tanh output 'a', d(a)/d(pre_activation) = 1 - a^2.
        # So the "error" on pre-activation is effectively scaled by (1 - a^2).
        # The "advantage" or "target direction" is match_reward.
        # We want to make the output values that led to good reward more likely.
        
        # Gradient of a pseudo-loss (-Reward * sum(y_full_activated)) w.r.t pre_activations:
        # d(-R * y_full)/d(y_pre) = -R * (1 - y_full^2)
        # This reinforces positive actions if R is positive, and negative actions if R is positive. This seems wrong.
        # It should be: if R > 0, make action more like itself. If R < 0, make action less like itself.
        # Let error_signal_on_output_activations = match_reward * y_full_activated (as in original)

        error_signal_on_output_activations = match_reward * y_full_activated

        # Gradient for the output layer's pre-activation
        # delta_L / delta_z_out = (delta_L / delta_y_out) * (delta_y_out / delta_z_out)
        # delta_y_out / delta_z_out = (1 - y_full_activated**2)
        delta_output_layer_pre_activation = error_signal_on_output_activations * (1 - y_full_activated**2)

        # Gradient of Loss w.r.t. w_out
        dW_out = np.outer(delta_output_layer_pre_activation, h_activated)

        # Propagate error to hidden layer
        error_hidden_layer_activation = self.w_out.T @ delta_output_layer_pre_activation

        # Gradient for the hidden layer's pre-activation
        delta_hidden_layer_pre_activation = error_hidden_layer_activation * (1 - h_activated**2)

        # Gradient of Loss w.r.t. w_in
        dW_in = np.outer(delta_hidden_layer_pre_activation, x_input)

        return dW_in, dW_out

    def update_weights(self, dW_in, dW_out, learning_rate):
        """Updates weights using the calculated gradients."""
        self.w_in -= learning_rate * dW_in
        self.w_out -= learning_rate * dW_out

    def mutate(self, mutation_rate_weights=1.0): # Sigma is now internal
        """Mutates the network's sigma and then its weights. Returns a new mutated TinyNet."""
        # 1. Mutate sigma (self-adaptation)
        # sigma' = sigma * exp(tau * N(0,1))
        # tau is a learning rate for sigma. A common choice is 1/sqrt(2*sqrt(num_parameters))
        # or 1/sqrt(2*num_parameters). Let's use config.SIGMA_ADAPTATION_TAU.
        new_sigma = self.mutation_sigma * np.exp(config.SIGMA_ADAPTATION_TAU * self.rng.normal(0, 1))
        new_sigma = max(1e-5, new_sigma) # Prevent sigma from becoming too small or zero

        # 2. Mutate weights using the new_sigma
        w_in_mutated = self.w_in.copy()
        w_out_mutated = self.w_out.copy()

        # Mutate only a fraction of weights or all, based on mutation_rate_weights
        # For simplicity, let's assume mutation_rate_weights applies to whether mutation happens at all.
        # The prompt implies sigma is mutated, then weights.
        if self.rng.random() < mutation_rate_weights: # This could be a per-weight mutation prob too
            noise_in = self.rng.normal(0, new_sigma, self.w_in.shape).astype(np.float64)
            w_in_mutated += noise_in

        if self.rng.random() < mutation_rate_weights: # Separate chance for w_out
            noise_out = self.rng.normal(0, new_sigma, self.w_out.shape).astype(np.float64)
            w_out_mutated += noise_out
        
        mutated_net = TinyNet(w_in_mutated, w_out_mutated,
                              self.input_size, self.hidden_size, self.output_size,
                              rng=self.rng, initial_sigma=new_sigma) # Pass new sigma
        mutated_net.fitness = 0.0 # New mutant has no fitness yet
        return mutated_net


    @classmethod
    def crossover(cls, parent1, parent2, rng_instance=None):
        """Performs uniform crossover between two parent TinyNets."""
        if rng_instance is None: # Should be passed from EvolutionOrchestrator
            rng_instance = np.random.default_rng()

        # Ensure parents have compatible shapes (especially important if network structure can vary)
        if not (parent1.w_in.shape == parent2.w_in.shape and \
                parent1.w_out.shape == parent2.w_out.shape and \
                parent1.input_size == parent2.input_size and \
                parent1.hidden_size == parent2.hidden_size and \
                parent1.output_size == parent2.output_size):
            # Fallback: return a copy of parent1 if dimensions mismatch (or raise error)
            # This shouldn't happen if population has uniform structure.
            print("Warning: Crossover between parents with different NN structures attempted. Returning copy of parent1.")
            # Create a new instance to avoid aliasing
            return cls(parent1.w_in.copy(), parent1.w_out.copy(),
                       parent1.input_size, parent1.hidden_size, parent1.output_size,
                       rng_instance, parent1.mutation_sigma)


        mask_in = rng_instance.random(parent1.w_in.shape) < 0.5
        w_in_child = np.where(mask_in, parent1.w_in, parent2.w_in)

        mask_out = rng_instance.random(parent1.w_out.shape) < 0.5
        w_out_child = np.where(mask_out, parent1.w_out, parent2.w_out)

        # Sigma for child: take from parent1 (or average, or random choice)
        child_sigma = parent1.mutation_sigma # Simple inheritance

        child_net = cls(w_in_child, w_out_child,
                        parent1.input_size, parent1.hidden_size, parent1.output_size,
                        rng_instance, child_sigma)
        child_net.fitness = 0.0 # Child has no fitness yet
        return child_net


    def get_genome_params(self):
        """Returns parameters that define the genome for saving/loading."""
        return self.w_in, self.w_out, self.mutation_sigma