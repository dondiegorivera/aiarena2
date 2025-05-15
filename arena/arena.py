# evo_arena/arena/arena.py
import pygame
import math
from agents.body import AgentBody
# from agents.brain import TinyNet # Not strictly needed here, only for type hint if used
import config # Import config

class Arena:
    def __init__(self, width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT,
                 wall_bounce_loss_factor=config.WALL_BOUNCE_LOSS_FACTOR, rng=None):
        self.width = float(width)
        self.height = float(height)
        self.agents = []
        self.wall_bounce_loss_factor = float(wall_bounce_loss_factor)
        self.game_time = 0.0
        self.rng = rng if rng is not None else np.random.default_rng() # Fallback, but should be passed

    def add_agent(self, agent):
        # AgentBody constructor now handles initial_x/y etc.
        self.agents.append(agent)

    def get_alive_agents(self):
        return [agent for agent in self.agents if agent.is_alive()]

    def get_agent_by_id(self, agent_id):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def update(self, dt):
        self.game_time += dt

        # Agents decide actions based on current state (before any movement this tick)
        # Create a stable list of agent states for this tick's decisions
        # AgentBody.update will call get_inputs internally using this list.
        current_all_agents_for_perception = list(self.agents)

        for agent in self.agents:
            if agent.is_alive():
                agent.update(dt, self.width, self.height, current_all_agents_for_perception)
            else: # Ensure dead agents don't move or fire
                agent.vx = 0
                agent.vy = 0
                agent.is_firing_command = False


        # Process firing and hit detection
        for idx, firing_agent in enumerate(self.agents):
            if not firing_agent.is_alive() or not firing_agent.is_firing_command:
                continue

            # Reset firing command after checking it once per update cycle,
            # so it doesn't persist multiple ticks from one NN output.
            # This depends on desired behavior: if fire means "fire one shot then cool down"
            # or "keep firing as long as signal is high".
            # Current AgentBody.perform_actions_from_outputs sets it if cooldown allows.
            # Let's assume firing_command is for this tick only if it leads to a shot.
            # If a shot is fired, cooldown timer starts.
            # If it's not fired (e.g. target out of range), should it still be reset?
            # The "is_firing_command" is more like "wants_to_fire_this_tick".

            for target_idx, target_agent in enumerate(self.agents):
                if idx == target_idx or not target_agent.is_alive():
                    continue
                # No friendly fire if team_id is same and non-zero (FFA can have team_id 0 or all different)
                if firing_agent.team_id == target_agent.team_id and firing_agent.team_id != 0:
                    continue

                dx = target_agent.x - firing_agent.x
                dy = target_agent.y - firing_agent.y
                distance_sq = dx*dx + dy*dy

                effective_range = firing_agent.weapon_range + target_agent.radius # Hit if center within range+radius

                if distance_sq <= effective_range * effective_range:
                    if distance_sq < 1e-6 : continue # Avoid division by zero if exactly on top

                    angle_to_target_rad = math.atan2(dy, dx)
                    agent_facing_rad = math.radians(firing_agent.angle_deg)
                    # Calculate relative angle correctly (normalized to -pi, pi)
                    relative_angle_rad = (angle_to_target_rad - agent_facing_rad + math.pi) % (2 * math.pi) - math.pi
                    weapon_half_arc_rad = math.radians(firing_agent.weapon_arc_deg / 2.0)

                    if abs(relative_angle_rad) <= weapon_half_arc_rad:
                        damage_amount = firing_agent.weapon_damage
                        target_agent.take_damage(damage_amount)
                        firing_agent.damage_dealt += damage_amount # Track damage dealt by firer
            
            # After processing all targets for this firing_agent, if they fired (is_firing_command was true),
            # their cooldown would have been set in AgentBody.perform_actions.
            # We can reset is_firing_command here to ensure it's a per-tick decision,
            # or let perform_actions_from_outputs handle it based on the NN signal.
            # perform_actions_from_outputs sets it true if fire_signal > threshold AND cooldown <= 0.
            # This seems fine. If it was true and cooldown was <=0, it's now >0.
            # If cooldown was already >0, is_firing_command was set false by perform_actions.
            # So, no explicit reset needed here.

        # Physics (movement) and Wall Collisions (after agent.update has set vx, vy)
        for agent in self.agents:
            if not agent.is_alive():
                continue # Dead agents don't move

            # Apply wall collisions
            # Agent's x,y is center. Collision if x - radius < 0 or x + radius > width.
            if agent.x - agent.radius < 0:
                agent.x = agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
            elif agent.x + agent.radius > self.width:
                agent.x = self.width - agent.radius
                agent.vx *= -self.wall_bounce_loss_factor

            if agent.y - agent.radius < 0:
                agent.y = agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
            elif agent.y + agent.radius > self.height:
                agent.y = self.height - agent.radius
                agent.vy *= -self.wall_bounce_loss_factor


    def draw_bounds(self, screen):
        if 'pygame' in globals() and screen is not None:
             pygame.draw.rect(screen, (50, 50, 50), (0, 0, int(self.width), int(self.height)), 2)


    def check_match_end_conditions(self, max_duration_seconds=config.MATCH_DURATION_SECONDS):
        """Checks if the match should end."""
        alive_agents = self.get_alive_agents()

        if not alive_agents:
            return True, None, "All agents eliminated" # Could be a draw if simultaneous elimination

        # Timeout check
        if self.game_time >= max_duration_seconds:
            # Determine winner by HP or other tie-breaker if needed, or just draw
            teams_alive_count = {}
            for agent in alive_agents:
                teams_alive_count[agent.team_id] = teams_alive_count.get(agent.team_id, 0) + 1
            
            if len(teams_alive_count) > 1: # Multiple teams still alive
                return True, None, f"Timeout ({self.game_time:.1f}s). Draw (multiple teams)."
            elif len(teams_alive_count) == 1: # One team left
                winner_team_id = list(teams_alive_count.keys())[0]
                return True, winner_team_id, f"Timeout. Team {winner_team_id} wins (last standing)."
            else: # No teams identified (e.g. all team_id 0 and multiple alive)
                 return True, None, f"Timeout ({self.game_time:.1f}s). Draw (complex)."


        # Last team standing check
        teams_present = set(agent.team_id for agent in alive_agents)
        if len(teams_present) == 1:
            winner_team_id = list(teams_present)[0]
            # Check if it's team_id 0 (FFA scenario), then it's last individual
            if winner_team_id == 0 and len(alive_agents) > 1: # FFA, but multiple agents of team 0 alive
                return False, None, "Match ongoing" # Match continues until one agent of team 0 is left
            return True, winner_team_id, f"Team {winner_team_id} is the last one standing!"

        return False, None, "Match ongoing"


    def reset_arena_and_agents(self):
        """Resets arena time and all agents currently in the arena to their initial states."""
        self.game_time = 0.0
        for agent in self.agents:
            # AgentBody.reset_state is now more comprehensive
            agent.reset_state(agent.initial_x, agent.initial_y, agent.initial_angle_deg, agent.max_hp_initial)


    def run_match(self, agent_configs, max_duration_steps, dt):
        """
        Runs a single headless match.
        agent_configs: list of dicts, each with 'brain', 'team_id', 'agent_id', 'start_pos', 'hp'.
        """
        self.agents = [] # Clear previous agents
        self.game_time = 0.0

        common_agent_params = {
            'base_speed': config.AGENT_BASE_SPEED,
            'rotation_speed_dps': config.AGENT_ROTATION_SPEED_DPS,
            'radius': config.AGENT_RADIUS,
            'weapon_range': config.WEAPON_RANGE,
            'weapon_arc_deg': config.WEAPON_ARC_DEG,
            'weapon_cooldown_time': config.WEAPON_COOLDOWN_TIME,
            'weapon_damage': config.WEAPON_DAMAGE,
            'cooldown_jitter_factor': config.COOLDOWN_JITTER_FACTOR,
            'rng': self.rng, # Pass arena's RNG to agents
            'use_rnn': config.USE_RNN,
            'rnn_hidden_size': config.RNN_HIDDEN_SIZE
        }

        for i, agent_conf in enumerate(agent_configs):
            start_x, start_y, start_angle = agent_conf['start_pos']
            agent = AgentBody(
                x=start_x, y=start_y, angle_deg=start_angle,
                agent_id=agent_conf.get('agent_id', f"match_agent_{i+1}"),
                team_id=agent_conf.get('team_id', i+1),
                hp=agent_conf.get('hp', config.DEFAULT_AGENT_HP),
                brain=agent_conf.get('brain'),
                color=agent_conf.get('color', (50 + i*50, 50 + i*50, 150 - i*50)), # Basic unique color
                **common_agent_params
            )
            self.add_agent(agent) # AgentBody constructor sets initial_x/y etc.

        match_over = False
        winner_team_id = None
        end_message = "Match did not conclude properly."
        actual_steps = 0

        for step in range(max_duration_steps):
            actual_steps = step + 1
            self.update(dt)

            max_total_duration_seconds = max_duration_steps * dt
            match_over, winner_team_id, end_message = self.check_match_end_conditions(max_total_duration_seconds)
            if match_over:
                break

        if not match_over: # If loop finished by exhausting steps
            max_total_duration_seconds = max_duration_steps * dt
            is_final_over, final_winner_id, final_message = self.check_match_end_conditions(max_total_duration_seconds)
            winner_team_id = final_winner_id
            end_message = final_message
            if not is_final_over: # Should be caught by timeout as a draw
                end_message = f"Max steps ({max_duration_steps}) reached. Considered Draw by step limit."


        final_agents_state_info = []
        for agent in self.agents:
            final_agents_state_info.append({
                'agent_id': agent.agent_id,
                'team_id': agent.team_id,
                'hp': agent.hp,
                'is_alive': agent.is_alive(),
                'damage_dealt': agent.damage_dealt, # Include damage dealt for fitness
                'duration_steps': actual_steps if agent.is_alive() else agent.death_tick if hasattr(agent, 'death_tick') else actual_steps # Needs agent.death_tick to be set
            })
            # For simplicity, all agents get `actual_steps` if alive, or if death_tick isn't tracked.
            # EvolutionOrchestrator will use `actual_steps` from the main dict for survivors.

        return {
            'winner_team_id': winner_team_id,
            'duration_steps': actual_steps, # Total steps the match ran
            'game_time_at_end': round(self.game_time, 3),
            'end_message': end_message,
            'agents_final_state': final_agents_state_info
        }