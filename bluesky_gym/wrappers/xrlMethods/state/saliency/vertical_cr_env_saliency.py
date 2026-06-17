import gymnasium as gym
import numpy as np
import pygame
import bluesky as bs
import os
from bluesky_gym.envs.vertical_cr_env import ACTION_FREQUENCY, NUM_INTRUDERS, NM2KM, INTRUSION_DISTANCE, DISTANCE_MARGIN, VERTICAL_MARGIN
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper

class SaliencyVerticalControl(SaliencyMapV1Wrapper):
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped", model=None, xrl_rendering=True):
        pygame.font.init()
        super().__init__(env, safe_vals, debug, export_gifs_path, fps, color_mode, model, xrl_rendering=xrl_rendering)
        self.action_frequency = ACTION_FREQUENCY
        
    def reset(self, seed=None, options=None):
        obs, inf = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        if self.export_gifs_path is not None:
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
        if self.render_mode == "human":
            self._render_frame()
        return obs, inf
            
    def step(self, action, shap_values=None):
        self.unwrapped._get_action(action)
        self.last_action = action
        self.step_counter += 1
        self.frame_saved = False

        for i in range(self.action_frequency):
            if self.render_mode == "human":
                self._render_frame(shap_values=shap_values)
            bs.sim.step()

        observation = self.unwrapped._get_obs()
        reward, terminated = self.unwrapped._get_reward()
        info = self.unwrapped._get_info()

        if terminated:
            self.export_episode_gif()

        return observation, reward, terminated, False, info
    
    def _render_frame(self, shap_values=None):
        self._pre_render()
        
        max_distance = 180
        max_alt = 5000
        zero_offset = 25
        
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))

        # Ground
        pygame.draw.rect(canvas, (154,205,50), pygame.Rect((0, self.unwrapped.window_height-50), (self.unwrapped.window_width, 50)))
        
        # Target altitude
        target_alt = int((-1*(self.unwrapped.target_alt-max_alt)/max_alt)*(self.unwrapped.window_height-50))
        pygame.draw.line(canvas, (255,255,255), (0, target_alt), (self.unwrapped.window_width, target_alt))

        # Runway
        runway_start = int(((self.unwrapped.runway_distance + zero_offset)/max_distance)*self.unwrapped.window_width)
        runway_end = int(runway_start + (30/max_distance)*self.unwrapped.window_width)
        pygame.draw.line(canvas, (119,136,153), (runway_start, self.unwrapped.window_height-50), (runway_end, self.unwrapped.window_height-50), width=3)

        # Ownship
        aircraft_alt = int((-1*(self.unwrapped.altitude-max_alt)/max_alt)*(self.unwrapped.window_height-50))
        aircraft_start = int(((zero_offset)/max_distance)*self.unwrapped.window_width)
        aircraft_end = int(aircraft_start + (4/max_distance)*self.unwrapped.window_width)
        pygame.draw.line(canvas, (0,0,0), (aircraft_start, aircraft_alt), (aircraft_end, aircraft_alt), width=5)

        # Intruders with SHAP Injection
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_alt = int((-1*(bs.traf.alt[int_idx]-max_alt)/max_alt)*(self.unwrapped.window_height-50))
            int_x_dis = self.unwrapped.intruder_distance[int_idx-1] * self.unwrapped.cos_bearing[int_idx-1]
            int_y_dis = self.unwrapped.intruder_distance[int_idx-1] * self.unwrapped.sin_bearing[int_idx-1]
            
            width_temp = int(5 + int_y_dis/20)
            int_start = int(((zero_offset + int_x_dis)/max_distance)*self.unwrapped.window_width)
            int_end = int(int_start + (4/max_distance)*self.unwrapped.window_width)

            # --- XRL Coloring Logic ---
            if shap_values is not None and self.xrl_rendering:
                if i < len(shap_values.values[0]):
                    # Extract single feature influence
                    color = self._get_saliency_color(shap_values.values[0][i], np.max(np.abs(shap_values.values)), shap_values.base_values[0][0])
                else:
                    color = (80,80,80)
            else:
                color = (255,255,255) if abs(int_y_dis) > DISTANCE_MARGIN else (255,0,0)

            pygame.draw.line(canvas, color, (int_start, int_alt), (int_end, int_alt), width=width_temp)

            # Bounding Box
            hor_margin = (DISTANCE_MARGIN*NM2KM/max_distance)*self.unwrapped.window_width
            ver_margin = (VERTICAL_MARGIN/max_alt)*self.unwrapped.window_height
            pygame.draw.line(canvas, 'black', (int_start-hor_margin/2, int_alt-ver_margin), (int_end+hor_margin/2, int_alt-ver_margin), width=1)
            pygame.draw.line(canvas, 'black', (int_start-hor_margin/2, int_alt+ver_margin), (int_end+hor_margin/2, int_alt+ver_margin), width=1)
            pygame.draw.line(canvas, 'black', (int_start-hor_margin/2, int_alt-ver_margin), (int_start-hor_margin/2, int_alt+ver_margin), width=1)
            pygame.draw.line(canvas, 'black', (int_end+hor_margin/2, int_alt-ver_margin), (int_end+hor_margin/2, int_alt+ver_margin), width=1)

        # --- XRL Bar Logic ---
        if shap_values is not None and self.xrl_rendering:
            legend_x = 30
            legend_y = 40
            legend_width = 150  # Maps to bar_length argument
            legend_height = 20  # Maps to thickness argument
            shap_sums = [float(np.sum(shap_values.values))]
            
            # Extract expected base value and actual scalar action command
            base_val = shap_values.base_values[0][0]
            actual_act = self.last_action[0] if hasattr(self.last_action, "__len__") else self.last_action

            self._draw_shap_bar(
                canvas, 
                shap_sums[0], 
                legend_x, 
                legend_y, 
                legend_width, 
                legend_height, 
                "vertical", 
                "Climb", 
                "Descend", 
                "Overall Vertical Influence",
                base_value=base_val,
                actual_action=actual_act
            )

        if not self.frame_saved and self.export_gifs_path is not None and shap_values is not None:
            self._create_shap_row(shap_values.values[0])

        self._post_render(canvas)
