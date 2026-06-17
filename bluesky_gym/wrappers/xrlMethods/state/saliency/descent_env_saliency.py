import gymnasium as gym
import numpy as np
import pygame
import os
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper
import bluesky as bs
from bluesky_gym.envs.descent_env import ACTION_FREQUENCY

class SaliencyDescentControl(SaliencyMapV1Wrapper):
    def __init__(self, env, fps=5, model=None, xrl_rendering=True, export_gifs_path=None):
        pygame.font.init()
        # safe_vals is None as KernelSHAP uses empirical backgrounds
        super().__init__(env, None, False, export_gifs_path, fps, "clipped", model, xrl_rendering=xrl_rendering)
        self.hud_font = pygame.font.SysFont("monospace", 16, bold=True)
        self.episode_counter = 0

    def step(self, action, shap_values=None):
        self.unwrapped._get_action(action)
        self.last_action = action
        self.step_counter += 1
        
        # Reset flag to allow base class _post_render to save the new frame
        self.frame_saved = False

        for i in range(ACTION_FREQUENCY):
            if self.render_mode == "human":
                self._render_frame(shap_values=shap_values)
            bs.sim.step()

        observation = self.unwrapped._get_obs()
        reward, terminated = self.unwrapped._get_reward()
        info = self.unwrapped._get_info()

        # Trigger GIF export on episode termination
        if terminated and self.export_gifs_path is not None:
            self.export_episode_gif()

        return observation, reward, terminated, False, info
        
    def _get_diverging_color(self, phi, max_phi):
        """Returns Red for Climb (>0), Blue for Descend (<0) based on SHAP magnitude."""
        norm = min(1.0, abs(phi) / (max_phi + 1e-6))

        if isinstance(norm, (list, np.ndarray)):
            norm = norm[0]

        if phi > 0:
            return (255, int(255*(1-norm)), int(255*(1-norm)))
        else:
            return (int(255*(1-norm)), int(255*(1-norm)), 255)

    def _render_frame(self, shap_values=None):
        self._pre_render()
        
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))

        width = self.unwrapped.window_width
        height = self.unwrapped.window_height
        max_distance = 180
        max_alt = 5000
        zero_offset = 45

        # Feature Data Extraction
        phi = [0, 0, 0, 0]
        max_phi = 1.0
        if shap_values is not None and self.xrl_rendering:
            phi = shap_values.values[0]
            phi = [phi[0][0], phi[1][0], phi[2][0], phi[3][0]]
            max_phi = np.max(np.abs(phi))
            
        c_alt = self._get_diverging_color(phi[0], max_phi)
        c_vz = self._get_diverging_color(phi[1], max_phi)
        c_talt = self._get_diverging_color(phi[2], max_phi)
        c_dist = self._get_diverging_color(phi[3], max_phi)

        # Ground
        pygame.draw.rect(canvas, (154,205,50), pygame.Rect((0, height-50), (width, 50)))

        # Target Altitude (Colored by phi_target_altitude)
        target_alt_y = int((-1*(self.unwrapped.target_alt-max_alt)/max_alt)*(height-50))
        pygame.draw.line(canvas, c_talt, (0, target_alt_y), (width, target_alt_y), width=3)

        # Runway Distance (Colored by phi_runway_distance)
        runway_start = int(((self.unwrapped.runway_distance + zero_offset)/max_distance)*width)
        runway_end = int(runway_start + (30/max_distance)*width)
        pygame.draw.line(canvas, c_dist, (runway_start, height-50), (runway_end, height-50), width=8)

        # Aircraft Coordinates
        aircraft_alt_y = int((-1*(self.unwrapped.altitude-max_alt)/max_alt)*(height-50))
        aircraft_start_x = int(((zero_offset)/max_distance)*width)
        aircraft_end_x = int(aircraft_start_x + (4/max_distance)*width)
        
        # Altitude Projection Line (Colored by phi_altitude)
        pygame.draw.line(canvas, c_alt, (aircraft_start_x+2, aircraft_alt_y), (aircraft_start_x+2, height-50), width=2)
        
        # Aircraft Body
        pygame.draw.line(canvas, (0,0,0), (aircraft_start_x, aircraft_alt_y), (aircraft_end_x, aircraft_alt_y), width=5)

        # Standalone VSI Chevron (Colored by phi_vz)
        chevron_x = aircraft_end_x + 15
        chevron_size = 6
        if self.unwrapped.vz > 0: # Pointing Up
            points = [(chevron_x - chevron_size, aircraft_alt_y + chevron_size), 
                      (chevron_x + chevron_size, aircraft_alt_y + chevron_size), 
                      (chevron_x, aircraft_alt_y - chevron_size)]
        else: # Pointing Down
            points = [(chevron_x - chevron_size, aircraft_alt_y - chevron_size), 
                      (chevron_x + chevron_size, aircraft_alt_y - chevron_size), 
                      (chevron_x, aircraft_alt_y + chevron_size)]
        pygame.draw.polygon(canvas, c_vz, points)

        # --- XRL Overlays ---
        if shap_values is not None and self.xrl_rendering:
            base_val = shap_values.base_values[0][0]
            actual_act = float(self.last_action[0] if hasattr(self.last_action, "__len__") else self.last_action)
            
            # HUD Overlay
            hud_bg = pygame.Surface((280, 160))
            hud_bg.set_alpha(200)
            hud_bg.fill((255, 255, 255))
            canvas.blit(hud_bg, (width - 290, 10))

            labels = [
                f"Alt  : {self.unwrapped.altitude:.0f} m | phi: {phi[0]:+.2f}",
                f"Vz   : {self.unwrapped.vz:.0f} m/s | phi: {phi[1]:+.2f}",
                f"T-Alt: {self.unwrapped.target_alt:.0f} m | phi: {phi[2]:+.2f}",
                f"Dist : {self.unwrapped.runway_distance:.0f} km | phi: {phi[3]:+.2f}",
                f"Base : {base_val:+.2f} | Act: {actual_act:+.2f}"
            ]
            for idx, text in enumerate(labels):
                color = (0,0,0)
                if "phi" in text:
                    val = phi[idx]
                    color = (200,0,0) if val > 0 else (0,0,200) if val < 0 else (0,0,0)
                txt_surface = self.hud_font.render(text, True, color)
                canvas.blit(txt_surface, (width - 280, 20 + idx*25))

            # Vertical Hybrid Force Plot (Left Side)
            bar_len = 200
            bar_thick = 20
            bar_x = 25 
            bar_y = 50
            
            b_val = max(-1.0, min(1.0, base_val))
            a_val = max(-1.0, min(1.0, actual_act))
            
            # Unified Gradient Background
            self._draw_gradient_bar(canvas, (bar_x, bar_y), bar_len, bar_thick, horizontal=False)
            
            y_base = bar_y + int(((1.0 - b_val) / 2.0) * bar_len)
            y_act = bar_y + int(((1.0 - a_val) / 2.0) * bar_len)
            
            # Draw Unification Arrow (Left Side)
            arrow_x = bar_x - 12
            if abs(y_act - y_base) > 2:
                pygame.draw.line(canvas, (50, 50, 50), (arrow_x, y_base), (arrow_x, y_act), 2)
                arrow_size = 6
                if y_act > y_base:
                    pts = [(arrow_x - arrow_size, y_act - arrow_size), (arrow_x + arrow_size, y_act - arrow_size), (arrow_x, y_act)]
                else:
                    pts = [(arrow_x - arrow_size, y_act + arrow_size), (arrow_x + arrow_size, y_act + arrow_size), (arrow_x, y_act)]
                pygame.draw.polygon(canvas, (50, 50, 50), pts)
            
            # Stack SHAP contributions (Right Side of Bar)
            stack_x = bar_x + bar_thick + 2
            stack_thick = 10
            current_y = y_base
            
            for p in phi:
                if abs(p) < 0.01: continue
                dy = int((-p / 2.0) * bar_len) 
                target_y = current_y + dy
                
                # Constrain drawing bounds to prevent overflow
                draw_y1 = max(bar_y, min(bar_y + bar_len, current_y))
                draw_y2 = max(bar_y, min(bar_y + bar_len, target_y))
                
                rect_y = min(draw_y1, draw_y2)
                rect_h = abs(draw_y1 - draw_y2)
                
                c = (200,0,0) if p > 0 else (0,0,200)
                if rect_h > 0:
                    pygame.draw.rect(canvas, c, (stack_x, rect_y, stack_thick, rect_h))
                current_y = target_y
            
            # Draw Labels and Markers
            pygame.draw.line(canvas, (0,150,0), (bar_x - 5, y_base), (bar_x + bar_thick + 40, y_base), 3)
            canvas.blit(self.hud_font.render("Base", True, (0,150,0)), (bar_x + bar_thick + 45, y_base - 8))
            
            pygame.draw.line(canvas, (0,0,0), (bar_x - 5, y_act), (bar_x + bar_thick + 40, y_act), 4)
            canvas.blit(self.hud_font.render("Actual", True, (0,0,0)), (bar_x + bar_thick + 45, y_act - 8))
            
            canvas.blit(self.hud_font.render("Climb", True, (0,0,0)), (bar_x - 10, bar_y - 25))
            canvas.blit(self.hud_font.render("Descend", True, (0,0,0)), (bar_x - 20, bar_y + bar_len + 5))

        self._post_render(canvas)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        
        # Initialize directory for the current episode's frames
        if self.export_gifs_path is not None:
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
            
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, info
