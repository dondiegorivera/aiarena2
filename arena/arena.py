# evo_arena/arena/arena.py
import pygame
import math
from agents.body import AgentBody
import config # Import config
import numpy as np # For rng fallback

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
        current_all_agents_for_perception = list(self.agents)

        for agent in self.agents:
            if agent.is_alive():
                agent.update(dt, self.width, self.height, current_all_agents_for_perception)
            else: 
                agent.vx = 0
                agent.vy = 0
                agent.is_firing_command = False

        for idx, firing_agent in enumerate(self.agents):
            if not firing_agent.is_alive() or not firing_agent.is_firing_command:
                continue
            
            for target_idx, target_agent in enumerate(self.agents):
                if idx == target_idx or not target_agent.is_alive():
                    continue
                if firing_agent.team_id == target_agent.team_id and firing_agent.team_id != 0:
                    continue

                dx = target_agent.x - firing_agent.x
                dy = target_agent.y - firing_agent.y
                distance_sq = dx*dx + dy*dy
                effective_range = firing_agent.weapon_range + target_agent.radius 

                if distance_sq <= effective_range * effective_range:
                    if distance_sq < 1e-6 : continue 
                    angle_to_target_rad = math.atan2(dy, dx)
                    agent_facing_rad = math.radians(firing_agent.angle_deg)
                    relative_angle_rad = (angle_to_target_rad - agent_facing_rad + math.pi) % (2 * math.pi) - math.pi
                    weapon_half_arc_rad = math.radians(firing_agent.weapon_arc_deg / 2.0)

                    if abs(relative_angle_rad) <= weapon_half_arc_rad:
                        damage_amount = firing_agent.weapon_damage
                        target_agent.take_damage(damage_amount)
                        firing_agent.damage_dealt += damage_amount 
            
        for agent in self.agents:
            if not agent.is_alive():
                continue 

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
            return True, None, "All agents eliminated" 

        if self.game_time >= max_duration_seconds:
            teams_alive_count = {}
            for agent in alive_agents:
                teams_alive_count[agent.team_id] = teams_alive_count.get(agent.team_id, 0) + 1
            
            if len(teams_alive_count) > 1: 
                return True, None, f"Timeout ({self.game_time:.1f}s). Draw (multiple teams)."
            elif len(teams_alive_count) == 1: 
                winner_team_id = list(teams_alive_count.keys())[0]
                return True, winner_team_id, f"Timeout. Team {winner_team_id} wins (last standing)."
            else: 
                 return True, None, f"Timeout ({self.game_time:.1f}s). Draw (complex)."

        teams_present = set(agent.team_id for agent in alive_agents)
        if len(teams_present) == 1:
            winner_team_id = list(teams_present)[0]
            if winner_team_id == 0 and len(alive_agents) > 1: 
                return False, None, "Match ongoing" 
            return True, winner_team_id, f"Team {winner_team_id} is the last one standing!"

        return False, None, "Match ongoing"

    def reset_arena_and_agents(self):
        """Resets arena time and all agents currently in the arena to their initial states."""
        self.game_time = 0.0
        for agent in self.agents:
            agent.reset_state(agent.initial_x, agent.initial_y, agent.initial_angle_deg, agent.max_hp_initial)

    def run_match(self, agent_configs, max_duration_steps, dt):
        """
        Runs a single headless match.
        agent_configs: list of dicts, each with 'brain', 'team_id', 'agent_id', 'start_pos', 'hp'.
        """
        self.agents = [] 
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
            'rng': self.rng, 
            # 'use_rnn' and 'rnn_hidden_size' are NOT passed here anymore.
            # AgentBody.__init__ will infer them from the brain or global config.
        }

        for i, agent_conf in enumerate(agent_configs):
            start_x, start_y, start_angle = agent_conf['start_pos']
            agent = AgentBody(
                x=start_x, y=start_y, angle_deg=start_angle,
                agent_id=agent_conf.get('agent_id', f"match_agent_{i+1}"),
                team_id=agent_conf.get('team_id', i+1),
                hp=agent_conf.get('hp', config.DEFAULT_AGENT_HP),
                brain=agent_conf.get('brain'), # This brain object is crucial for AgentBody to infer RNN status
                color=agent_conf.get('color', (50 + i*50 % 205, 50 + i*70 % 205, 150 - i*60 % 205)),
                **common_agent_params # Pass all other common params
            )
            self.add_agent(agent) 

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

        if not match_over: 
            max_total_duration_seconds = max_duration_steps * dt
            is_final_over, final_winner_id, final_message = self.check_match_end_conditions(max_total_duration_seconds)
            winner_team_id = final_winner_id
            end_message = final_message
            if not is_final_over: 
                end_message = f"Max steps ({max_duration_steps}) reached. Considered Draw by step limit."


        final_agents_state_info = []
        for agent in self.agents:
            final_agents_state_info.append({
                'agent_id': agent.agent_id,
                'team_id': agent.team_id,
                'hp': agent.hp,
                'is_alive': agent.is_alive(),
                'damage_dealt': agent.damage_dealt, 
                'duration_steps': actual_steps 
            })

        return {
            'winner_team_id': winner_team_id,
            'duration_steps': actual_steps, 
            'game_time_at_end': round(self.game_time, 3),
            'end_message': end_message,
            'agents_final_state': final_agents_state_info
        }