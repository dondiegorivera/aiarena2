# evo_arena/ui/viewer.py
import pygame
import math
import config # Import config

class Viewer:
    def __init__(self, width, height, arena, title="Evo Arena"):
        pygame.init()
        self.width = int(width)
        self.height = int(height)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.arena = arena
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)
        self.info_font = pygame.font.SysFont(None, 24)
        self.is_running_flag = True # For visual match series control

    def is_running(self): # For match series control
        return self.is_running_flag

    def draw_firing_cone(self, screen, agent):
        if not agent.is_alive() or not agent.is_firing_command:
            return

        cone_color = (255, 255, 0, 100)  # Yellow, semi-transparent
        p1 = (int(agent.x), int(agent.y))

        angle_left_rad = math.radians(agent.angle_deg - agent.weapon_arc_deg / 2.0)
        p2_x = agent.x + agent.weapon_range * math.cos(angle_left_rad)
        p2_y = agent.y + agent.weapon_range * math.sin(angle_left_rad)
        p2 = (int(p2_x), int(p2_y))

        angle_right_rad = math.radians(agent.angle_deg + agent.weapon_arc_deg / 2.0)
        p3_x = agent.x + agent.weapon_range * math.cos(angle_right_rad)
        p3_y = agent.y + agent.weapon_range * math.sin(angle_right_rad)
        p3 = (int(p3_x), int(p3_y))

        # Use a Surface for alpha blending
        cone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surface, cone_color, [p1, p2, p3])
        screen.blit(cone_surface, (0,0))


    def render_frame(self, show_game_over_message=True, game_over_message_text=""):
        """Renders a single frame of the simulation, optionally with a game over message."""
        self.screen.fill((30, 30, 30))
        self.arena.draw_bounds(self.screen)

        for agent_to_draw in self.arena.agents:
            agent_to_draw.draw(self.screen)
            if agent_to_draw.is_firing_command and agent_to_draw.is_alive():
                self.draw_firing_cone(self.screen, agent_to_draw)
        
        if show_game_over_message and game_over_message_text:
            msg_surf = self.font.render(game_over_message_text, True, (255, 255, 0))
            msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(msg_surf, msg_rect)

            reset_surf = self.info_font.render("Press 'R' to reset, ESC to quit/skip", True, (200, 200, 200))
            reset_rect = reset_surf.get_rect(center=(self.width / 2, self.height / 2 + 40))
            self.screen.blit(reset_surf, reset_rect)
        
        # Moved HUD text to a separate method for match series
        # Time text is general
        time_text = self.info_font.render(f"Time: {self.arena.game_time:.1f}s", True, (220, 220, 220))
        self.screen.blit(time_text, (10, 10))


    def render_hud_texts(self, hp_texts=None, score_text=None, custom_texts=None):
        """Renders HUD elements like HP, score, etc. Called after render_frame normally."""
        if hp_texts:
            for i, (text, color) in enumerate(hp_texts):
                surf = self.info_font.render(text, True, color)
                self.screen.blit(surf, (10, self.height - 20 * (len(hp_texts) - i) ))
        
        if score_text:
            surf = self.font.render(score_text, True, (200, 200, 50))
            rect = surf.get_rect(centerx=self.width / 2, y=10)
            self.screen.blit(surf, rect)

        if custom_texts:
            for i, (text, color, y_offset) in enumerate(custom_texts):
                 surf = self.font.render(text, True, color)
                 rect = surf.get_rect(center=(self.width / 2, self.height / 2 + y_offset))
                 self.screen.blit(surf, rect)


    def run_simulation_loop(self, fps, manual_agent_id=None):
        """Main simulation loop for modes like manual play or showcase."""
        self.is_running_flag = True
        dt = 1.0 / fps
        game_over_text = ""
        game_is_over = False

        # Initial agent states are now set by AgentBody constructor or reset_state
        # Arena reset_arena_and_agents will use these.

        while self.is_running_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running_flag = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running_flag = False
                    if event.key == pygame.K_r and game_is_over:
                        self.arena.reset_arena_and_agents() # Resets agents to their initial positions
                        # If manual mode, ensure agent is placed correctly if initial_pos was different
                        # This simple reset is fine for showcase.
                        game_is_over = False
                        game_over_text = ""

            if not game_is_over:
                keys = pygame.key.get_pressed()
                if manual_agent_id:
                    manual_agent = self.arena.get_agent_by_id(manual_agent_id)
                    if manual_agent and not manual_agent.is_dummy and manual_agent.brain is None:
                        manual_agent.manual_control(keys)

                self.arena.update(dt)
                is_over_cond, _, message = self.arena.check_match_end_conditions(max_duration_seconds=config.MATCH_DURATION_SECONDS * 10) # Longer for showcase
                if is_over_cond:
                    game_is_over = True
                    game_over_text = message
                    print(game_over_text)

            self.render_frame(show_game_over_message=game_is_over, game_over_message_text=game_over_text)
            
            # Display HP for all agents in showcase/manual
            hp_display_list = []
            for ag in self.arena.agents:
                 hp_display_list.append( (f"{ag.agent_id} HP: {ag.hp:.0f}", ag.color) )
            self.render_hud_texts(hp_texts=hp_display_list)


            pygame.display.flip()
            self.clock.tick(fps)

        pygame.quit()


    def wait_for_keypress_to_continue(self, main_message, sub_message, prompt_message,
                                      continue_key=pygame.K_SPACE, abort_key=pygame.K_ESCAPE, is_final_screen=False):
        """Displays messages and waits for a key press to continue or abort."""
        if not pygame.display.get_init():
            self.is_running_flag = False
            return

        waiting = True
        while waiting and self.is_running_flag:
            self.screen.fill((30,30,30))
            
            texts_to_render = []
            if main_message: texts_to_render.append( (main_message, (255,255,0), -40) )
            if sub_message: texts_to_render.append( (sub_message, (200,200,200), 0) )
            if prompt_message: texts_to_render.append( (prompt_message, (180,180,180), 40) )
            
            self.render_hud_texts(custom_texts=texts_to_render)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False; self.is_running_flag = False
                if event.type == pygame.KEYDOWN:
                    if continue_key and event.key == continue_key:
                        waiting = False
                    if abort_key and event.key == abort_key:
                        waiting = False; self.is_running_flag = False
                        if is_final_screen: pygame.quit() # Quit pygame on final escape

            self.clock.tick(15) # Lower FPS for waiting screen