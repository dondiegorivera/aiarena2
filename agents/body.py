# evo_arena/agents/body.py
import pygame
import math
import numpy as np
# import random # No longer using global random, pass RNG or use rand.get_rng()
import config # Import config
# from rand import get_rng # Could use this if not passing rng explicitly

class AgentBody:
    def __init__(self, x, y, angle_deg, base_speed, rotation_speed_dps,
                 radius=config.AGENT_RADIUS, color=(0, 0, 255), agent_id="agent", team_id=0,
                 hp=config.DEFAULT_AGENT_HP, brain=None, is_dummy=False,
                 weapon_range=config.WEAPON_RANGE, weapon_arc_deg=config.WEAPON_ARC_DEG,
                 weapon_cooldown_time=config.WEAPON_COOLDOWN_TIME, weapon_damage=config.WEAPON_DAMAGE,
                 cooldown_jitter_factor=config.COOLDOWN_JITTER_FACTOR,
                 rng=None, # Expect an RNG instance
                 use_rnn=config.USE_RNN, rnn_hidden_size=config.RNN_HIDDEN_SIZE):

        self.agent_id = str(agent_id)
        self.team_id = int(team_id)
        self.x = float(x)
        self.y = float(y)
        self.angle_deg = float(angle_deg)
        self.radius = float(radius)
        self.color = color
        self.is_dummy = is_dummy
        self.brain = brain # Should be a TinyNet instance or None

        self.base_speed = float(base_speed)
        self.rotation_speed_dps = float(rotation_speed_dps)

        self.vx = 0.0
        self.vy = 0.0

        self.max_hp = float(hp)
        self.hp = float(hp)
        self.damage_dealt = 0.0 # For fitness tracking

        self.weapon_range = float(weapon_range)
        self.weapon_arc_deg = float(weapon_arc_deg)
        self.base_weapon_cooldown_time = float(weapon_cooldown_time)
        self.weapon_damage = float(weapon_damage)
        self.weapon_cooldown_timer = 0.0
        self.is_firing_command = False

        self._manual_thrust_forward = False
        self._manual_thrust_backward = False
        self._manual_strafe_left = False # Added for consistency with continuous
        self._manual_strafe_right = False# Added for consistency with continuous
        self._manual_rotate_left = False
        self._manual_rotate_right = False
        self._manual_fire = False

        self.max_abs_velocity_component = self.base_speed # Used for normalizing velocity inputs if needed

        self.cooldown_jitter_factor = float(cooldown_jitter_factor)
        self.rng = rng if rng is not None else np.random.default_rng() # Fallback, but should be passed

        # RNN related
        self.use_rnn = use_rnn
        self.rnn_hidden_size = rnn_hidden_size if self.use_rnn else 0
        if self.use_rnn:
            self.recurrent_hidden_state = np.zeros(self.rnn_hidden_size, dtype=np.float64)
        else:
            self.recurrent_hidden_state = None # Or an empty array if preferred by downstream logic

        # Store initial state for resets
        self.initial_x = self.x
        self.initial_y = self.y
        self.initial_angle_deg = self.angle_deg
        self.max_hp_initial = self.max_hp


    def get_effective_cooldown_time(self):
        """Calculates the actual cooldown time for the next shot, including jitter, using the provided RNG."""
        if self.cooldown_jitter_factor == 0:
            return self.base_weapon_cooldown_time

        jitter_range = self.base_weapon_cooldown_time * self.cooldown_jitter_factor
        jitter = self.rng.uniform(-jitter_range, jitter_range)
        effective_cooldown = self.base_weapon_cooldown_time + jitter
        return max(0.05, effective_cooldown) # Ensure cooldown doesn't become zero or negative

    def is_alive(self):
        return self.hp > 0

    def take_damage(self, amount):
        if not self.is_alive(): return
        self.hp -= amount
        if self.hp < 0:
            self.hp = 0
        # Damage dealt is tracked by the firer, not here.

    def manual_control(self, keys):
        if self.is_dummy or self.brain is not None:
            return

        self._manual_thrust_forward = keys[pygame.K_UP]
        self._manual_thrust_backward = keys[pygame.K_DOWN]
        # Assuming Q/E for strafe, A/D or Left/Right for rotate
        # This example uses Left/Right for rotate, no manual strafe yet.
        # To add manual strafe:
        # self._manual_strafe_left = keys[pygame.K_q]
        # self._manual_strafe_right = keys[pygame.K_e]
        self._manual_rotate_left = keys[pygame.K_LEFT]
        self._manual_rotate_right = keys[pygame.K_RIGHT]
        self._manual_fire = keys[pygame.K_SPACE]

    def _cast_ray(self, start_x, start_y, angle_rad, max_dist, all_agents, arena_width, arena_height):
        """Casts a single ray and returns the distance to the first collision (wall or agent)."""
        min_hit_dist = max_dist

        # Ray endpoint if no collision
        ray_end_x = start_x + max_dist * math.cos(angle_rad)
        ray_end_y = start_y + max_dist * math.sin(angle_rad)

        # 1. Check wall collisions
        # Top wall (y=0)
        if math.sin(angle_rad) < -1e-6: # Ray pointing somewhat upwards
            t = (0 - start_y) / math.sin(angle_rad)
            if 0 < t < min_hit_dist:
                hit_x = start_x + t * math.cos(angle_rad)
                if 0 <= hit_x <= arena_width:
                    min_hit_dist = t
        # Bottom wall (y=arena_height)
        if math.sin(angle_rad) > 1e-6: # Ray pointing somewhat downwards
            t = (arena_height - start_y) / math.sin(angle_rad)
            if 0 < t < min_hit_dist:
                hit_x = start_x + t * math.cos(angle_rad)
                if 0 <= hit_x <= arena_width:
                    min_hit_dist = t
        # Left wall (x=0)
        if math.cos(angle_rad) < -1e-6: # Ray pointing somewhat left
            t = (0 - start_x) / math.cos(angle_rad)
            if 0 < t < min_hit_dist:
                hit_y = start_y + t * math.sin(angle_rad)
                if 0 <= hit_y <= arena_height:
                    min_hit_dist = t
        # Right wall (x=arena_width)
        if math.cos(angle_rad) > 1e-6: # Ray pointing somewhat right
            t = (arena_width - start_x) / math.cos(angle_rad)
            if 0 < t < min_hit_dist:
                hit_y = start_y + t * math.sin(angle_rad)
                if 0 <= hit_y <= arena_height:
                    min_hit_dist = t
        
        # 2. Check agent collisions (line-circle intersection)
        for other_agent in all_agents:
            if other_agent is self or not other_agent.is_alive():
                continue

            # Vector from ray origin to circle center
            oc_x = other_agent.x - start_x
            oc_y = other_agent.y - start_y

            # Project OC onto ray direction vector D = (cos(angle_rad), sin(angle_rad))
            # t_ca = OC dot D
            t_ca = oc_x * math.cos(angle_rad) + oc_y * math.sin(angle_rad)

            if t_ca < 0 or t_ca > min_hit_dist : # Circle center is behind ray origin or further than current closest hit
                continue

            # Distance squared from circle center to closest point on ray line
            # d_sq = (OC dot OC) - t_ca^2
            d_sq = (oc_x**2 + oc_y**2) - t_ca**2
            
            radius_sq = other_agent.radius**2
            if d_sq > radius_sq: # Ray misses the circle
                continue

            # Half-chord distance squared
            t_hc_sq = radius_sq - d_sq
            if t_hc_sq < 0: # Should not happen if d_sq <= radius_sq
                 continue
            
            t_hc = math.sqrt(t_hc_sq)

            # Intersection distances along the ray
            t0 = t_ca - t_hc
            # t1 = t_ca + t_hc # Farther intersection point, not usually needed for first hit

            if t0 > 1e-6 and t0 < min_hit_dist: # Valid intersection closer than current min_hit_dist
                min_hit_dist = t0
                
        return min_hit_dist


    def get_sensory_inputs(self, arena_width, arena_height, all_agents):
        """Generates sensory inputs for the NN (Lidar, health, weapon status, bias)."""
        # Lidar inputs
        lidar_distances = np.full(config.LIDAR_NUM_RAYS, config.LIDAR_MAX_RANGE, dtype=np.float64)
        agent_angle_rad = math.radians(self.angle_deg)

        for i in range(config.LIDAR_NUM_RAYS):
            ray_angle_offset = (i / config.LIDAR_NUM_RAYS) * 2 * math.pi # Spread rays over 360 degrees
            current_ray_angle_world = agent_angle_rad + ray_angle_offset
            
            dist = self._cast_ray(self.x, self.y, current_ray_angle_world,
                                  config.LIDAR_MAX_RANGE, all_agents,
                                  arena_width, arena_height)
            lidar_distances[i] = dist

        # Normalize lidar distances (0 to LIDAR_MAX_RANGE -> -1 to 1, or 0 to 1)
        # Using 0 to 1 for distances: 0 = close, 1 = far/no hit
        # Or -1 to 1: (2 * (value / max_val) - 1)
        # Let's use 1.0 for max range, 0.0 for min range (contact)
        # So, (max_range - dist) / max_range perhaps? Or dist / max_range.
        # Project spec says "all normalised to [-1, 1]".
        # For distance: 0 is close, MAX_RANGE is far.
        # Normalized: (dist / MAX_RANGE) * 2 - 1.  So 0 -> -1, MAX_RANGE -> 1.
        normalized_lidar = (lidar_distances / config.LIDAR_MAX_RANGE) * 2.0 - 1.0
        normalized_lidar = np.clip(normalized_lidar, -1.0, 1.0)


        # Health input (normalized -1 to 1)
        # 0 HP -> -1, max_hp -> 1
        normalized_health = (self.hp / self.max_hp) * 2.0 - 1.0 if self.max_hp > 0 else -1.0
        normalized_health = np.clip(normalized_health, -1.0, 1.0)

        # Weapon ready input (1.0 if ready, -1.0 if cooling)
        weapon_ready_input = 1.0 if self.weapon_cooldown_timer <= 0 else -1.0

        # Bias input
        bias_input = 1.0

        sensory_inputs_list = list(normalized_lidar) + [normalized_health, weapon_ready_input, bias_input]
        return np.array(sensory_inputs_list, dtype=np.float64)


    def get_inputs_for_nn(self, arena_width, arena_height, all_agents):
        """
        Prepares the full input vector for the neural network, including sensory inputs
        and recurrent state if RNN is used.
        """
        sensory_part = self.get_sensory_inputs(arena_width, arena_height, all_agents)

        if self.use_rnn and self.recurrent_hidden_state is not None:
            # Ensure recurrent_hidden_state has the correct shape and type
            if self.recurrent_hidden_state.shape[0] != self.rnn_hidden_size:
                 # This indicates a mismatch, possibly re-initialize or error
                 self.recurrent_hidden_state = np.zeros(self.rnn_hidden_size, dtype=np.float64)

            return np.concatenate((sensory_part, self.recurrent_hidden_state))
        else:
            return sensory_part


    def perform_actions_from_outputs(self, nn_outputs_full, dt):
        """Interprets NN outputs (actions + new recurrent state) and applies them."""
        if not self.is_alive():
            self.vx, self.vy = 0, 0
            return

        # Split NN output into actions and new recurrent state
        action_outputs = nn_outputs_full[:config.NUM_ACTIONS]
        if self.use_rnn:
            self.recurrent_hidden_state = nn_outputs_full[config.NUM_ACTIONS:]

        # Continuous control mapping
        # output[0]: Thrust (tanh: -1 to 1)
        # output[1]: Strafe (tanh: -1 to 1)
        # output[2]: Rotate (tanh: -1 to 1)
        # output[3]: Fire (tanh: -1 to 1, thresholded)

        thrust_input = action_outputs[0]  # -1 (backward) to 1 (forward)
        strafe_input = action_outputs[1]  # -1 (right) to 1 (left)
        rotate_input = action_outputs[2]  # -1 (right) to 1 (left)
        fire_signal = action_outputs[3]

        # Rotation
        # Positive rotate_input for left turn, negative for right turn
        self.angle_deg += rotate_input * self.rotation_speed_dps * dt
        self.angle_deg %= 360.0

        # Movement (Thrust & Strafe)
        agent_angle_rad = math.radians(self.angle_deg)
        cos_angle = math.cos(agent_angle_rad)
        sin_angle = math.sin(agent_angle_rad)

        # Thrust component
        thrust_vx = thrust_input * self.base_speed * cos_angle
        thrust_vy = thrust_input * self.base_speed * sin_angle

        # Strafe component (perp to heading)
        # Positive strafe_input for left strafe
        strafe_vx = strafe_input * self.base_speed * config.STRAFE_SPEED_FACTOR * math.cos(agent_angle_rad - math.pi / 2)
        strafe_vy = strafe_input * self.base_speed * config.STRAFE_SPEED_FACTOR * math.sin(agent_angle_rad - math.pi / 2)
        
        self.vx = thrust_vx + strafe_vx
        self.vy = thrust_vy + strafe_vy

        # Optional: Cap total speed if combining thrust and strafe can exceed base_speed
        current_speed_sq = self.vx**2 + self.vy**2
        if current_speed_sq > self.base_speed**2 and self.base_speed > 0:
            scale = self.base_speed / math.sqrt(current_speed_sq)
            self.vx *= scale
            self.vy *= scale
            
        # Firing
        if fire_signal > config.FIRE_THRESHOLD:
            if self.weapon_cooldown_timer <= 0:
                self.is_firing_command = True
                self.weapon_cooldown_timer = self.get_effective_cooldown_time()
            else:
                self.is_firing_command = False # Still cooling down
        else:
            self.is_firing_command = False


    def update(self, dt, arena_width=None, arena_height=None, all_agents=None):
        if not self.is_alive():
            self.vx, self.vy = 0,0
            return

        if self.weapon_cooldown_timer > 0:
            self.weapon_cooldown_timer -= dt
            if self.weapon_cooldown_timer < 0:
                self.weapon_cooldown_timer = 0

        if self.brain:
            if arena_width is None or arena_height is None or all_agents is None:
                # Should not happen if arena logic is correct
                self.vx, self.vy = 0,0
                self.is_firing_command = False
            else:
                # 1. Get full NN input (sensory + recurrent if any)
                full_nn_input = self.get_inputs_for_nn(arena_width, arena_height, all_agents)
                # 2. Get NN output (actions + new recurrent state if any)
                nn_outputs_full = self.brain(full_nn_input) # Brain __call__ needs to handle this
                # 3. Perform actions and update recurrent state
                self.perform_actions_from_outputs(nn_outputs_full, dt)
        elif not self.is_dummy: # Manual control
            # Rotation
            if self._manual_rotate_left: self.angle_deg -= self.rotation_speed_dps * dt
            if self._manual_rotate_right: self.angle_deg += self.rotation_speed_dps * dt
            self.angle_deg %= 360.0

            # Movement (simplified for manual: only forward/backward, no strafe yet)
            manual_thrust_speed = 0.0
            if self._manual_thrust_forward: manual_thrust_speed = self.base_speed
            elif self._manual_thrust_backward: manual_thrust_speed = -self.base_speed * 0.5 # Slower backward

            angle_rad = math.radians(self.angle_deg)
            self.vx = manual_thrust_speed * math.cos(angle_rad)
            self.vy = manual_thrust_speed * math.sin(angle_rad)
            # Manual strafe could be added here if keys are mapped

            # Firing
            if self._manual_fire:
                if self.weapon_cooldown_timer <= 0:
                    self.is_firing_command = True
                    self.weapon_cooldown_timer = self.get_effective_cooldown_time()
                # else: no change, still cooling
            else:
                self.is_firing_command = False # Reset command if key not pressed
        else: # Dummy agent
            self.vx, self.vy = 0,0
            self.is_firing_command = False

        # Apply physics
        self.x += self.vx * dt
        self.y += self.vy * dt
        # is_firing_command is set by action logic, not reset here per tick unless specifically managed

    def draw(self, screen):
        # ... (drawing logic largely unchanged, but uses self.radius as float)
        body_color = self.color
        if not self.is_alive():
            body_color = (50, 50, 50)
        pygame.draw.circle(screen, body_color, (int(self.x), int(self.y)), int(self.radius))

        if not self.is_alive():
            return

        angle_rad = math.radians(self.angle_deg)
        end_x = self.x + self.radius * math.cos(angle_rad)
        end_y = self.y + self.radius * math.sin(angle_rad)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

        if self.max_hp > 0:
            hp_bar_width = self.radius * 1.5
            hp_bar_height = 5
            hp_bar_x = self.x - hp_bar_width / 2
            hp_bar_y = self.y - self.radius - hp_bar_height - 3

            current_hp_ratio = self.hp / self.max_hp
            current_hp_width = current_hp_ratio * hp_bar_width

            pygame.draw.rect(screen, (150,0,0), (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
            if current_hp_width > 0:
                pygame.draw.rect(screen, (0,200,0), (hp_bar_x, hp_bar_y, current_hp_width, hp_bar_height))

        if self.weapon_cooldown_timer > 0 and self.base_weapon_cooldown_time > 0:
            cooldown_ratio = self.weapon_cooldown_timer / self.base_weapon_cooldown_time
            cooldown_ratio = min(1.0, cooldown_ratio)

            arc_radius = self.radius * 0.6
            arc_rect = pygame.Rect(self.x - arc_radius, self.y - arc_radius, arc_radius*2, arc_radius*2)
            start_angle_disp = -math.pi/2 # Start at top for display
            end_angle_disp = start_angle_disp + (2 * math.pi * cooldown_ratio)
            try:
                 pygame.draw.arc(screen, (200,200,0), arc_rect, start_angle_disp, end_angle_disp, 2)
            except TypeError: # Can happen if angles are identical
                 pass


    def get_state_for_replay(self):
        return {
            'id': self.agent_id,
            'team_id': self.team_id,
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'angle_deg': round(self.angle_deg, 2),
            'hp': round(self.hp, 1),
            'is_alive': self.is_alive(),
            'is_firing_cmd': self.is_firing_command,
            'vx': round(self.vx, 2),
            'vy': round(self.vy, 2),
            'damage_dealt': round(self.damage_dealt, 1),
            # 'recurrent_state': self.recurrent_hidden_state.tolist() if self.use_rnn and self.recurrent_hidden_state is not None else None
        }

    def reset_state(self, x, y, angle_deg, hp=None):
        self.x = float(x)
        self.y = float(y)
        self.angle_deg = float(angle_deg)
        self.hp = float(hp if hp is not None else self.max_hp_initial) # Use stored initial max_hp
        self.vx = 0.0
        self.vy = 0.0
        self.weapon_cooldown_timer = 0.0
        self.is_firing_command = False
        self.damage_dealt = 0.0 # Reset damage dealt

        self._manual_thrust_forward = False
        self._manual_thrust_backward = False
        self._manual_strafe_left = False
        self._manual_strafe_right = False
        self._manual_rotate_left = False
        self._manual_rotate_right = False
        self._manual_fire = False

        if self.use_rnn:
            self.recurrent_hidden_state = np.zeros(self.rnn_hidden_size, dtype=np.float64)