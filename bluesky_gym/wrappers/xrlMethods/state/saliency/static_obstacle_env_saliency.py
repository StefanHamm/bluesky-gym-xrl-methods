import gymnasium as gym
import numpy as np
import pygame
import os
import bluesky as bs
from bluesky_gym.envs.static_obstacle_env import ACTION_FREQUENCY, MAX_DISTANCE, NM2KM
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper

class SaliencyStaticControl(SaliencyMapV1Wrapper):
    def __init__(self, env, fps=5, model=None, xrl_rendering=True, export_gifs_path=None):
        pygame.font.init()
        super().__init__(env, None, False, export_gifs_path, fps, "clipped", model, xrl_rendering=xrl_rendering)
        self.hud_font = pygame.font.SysFont("monospace", 14, bold=True)
        self.episode_counter = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        if self.export_gifs_path is not None:
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action, shap_values=None):
        self.unwrapped._get_action(action)
        self.last_action = action
        self.step_counter += 1
        self.frame_saved = False

        for i in range(ACTION_FREQUENCY):
            if self.render_mode == "human":
                self._render_frame(shap_values=shap_values)
            bs.sim.step()
            
            # Static Obstacle Env handles terminal states inside its sub-step
            reward, done, terminated = self.unwrapped._get_reward()
            if terminated or done:
                observation = self.unwrapped._get_obs()
                self.unwrapped.total_reward += reward
                info = self.unwrapped._get_info()
                if self.export_gifs_path is not None:
                    self.export_episode_gif()
                return observation, reward, done, terminated, info

        observation = self.unwrapped._get_obs()
        self.unwrapped.total_reward += reward
        info = self.unwrapped._get_info()
        return observation, reward, done, terminated, info

    def _get_diverging_color(self, phi, max_phi):
        """Returns Red (>0) or Blue (<0) based on SHAP magnitude."""
        norm = min(1.0, abs(phi) / (max_phi + 1e-6))
        if phi > 0:
            return (255, int(255*(1-norm)), int(255*(1-norm)))
        else:
            return (int(255*(1-norm)), int(255*(1-norm)), 255)

    def _draw_manual_bar(self, canvas, x, y, length, thickness, is_horizontal, base_val, actual_val, title, pos_label, neg_label):
        # Draw Background Gradient
        for i in range(length):
            ratio = i / length
            color = (int(255*(1-ratio)), 0, int(255*ratio)) if is_horizontal else (int(255*(1-ratio)), 0, int(255*ratio))
            if is_horizontal:
                pygame.draw.line(canvas, color, (x + i, y), (x + i, y + thickness))
            else:
                pygame.draw.line(canvas, color, (x, y + i), (x + thickness, y + i))
        pygame.draw.rect(canvas, (0,0,0), (x, y, length if is_horizontal else thickness, thickness if is_horizontal else length), 2)

        # Map values to pixels
        b_val = max(-1.0, min(1.0, base_val))
        a_val = max(-1.0, min(1.0, actual_val))

        if is_horizontal:
            px_base = x + int(((b_val + 1.0) / 2.0) * length)
            px_act = x + int(((a_val + 1.0) / 2.0) * length)
            
            # Draw Net Influence Block
            fill_x = min(px_base, px_act)
            fill_w = abs(px_act - px_base)
            fill_color = (200, 0, 0) if a_val > b_val else (0, 0, 200)
            if fill_w > 0:
                pygame.draw.rect(canvas, fill_color, (fill_x, y, fill_w, thickness))

            # Base Line (Green)
            pygame.draw.line(canvas, (0, 180, 0), (px_base, y - 5), (px_base, y + thickness + 5), 3)
            canvas.blit(self.hud_font.render("Base", True, (0, 180, 0)), (px_base - 15, y - 20))
            
            # Actual Line (Black)
            pygame.draw.line(canvas, (0, 0, 0), (px_act, y - 5), (px_act, y + thickness + 5), 4)
            canvas.blit(self.hud_font.render("Act", True, (0, 0, 0)), (px_act - 10, y + thickness + 8))
            
            # Labels
            canvas.blit(self.hud_font.render(title, True, (0,0,0)), (x, y - 40))
            canvas.blit(self.hud_font.render(neg_label, True, (0,0,0)), (x - 40, y + 2))
            canvas.blit(self.hud_font.render(pos_label, True, (0,0,0)), (x + length + 10, y + 2))

        else:
            px_base = y + int(((1.0 - b_val) / 2.0) * length)
            px_act = y + int(((1.0 - a_val) / 2.0) * length)
            
            # Draw Net Influence Block
            fill_y = min(px_base, px_act)
            fill_h = abs(px_act - px_base)
            fill_color = (200, 0, 0) if a_val > b_val else (0, 0, 200)
            if fill_h > 0:
                pygame.draw.rect(canvas, fill_color, (x, fill_y, thickness, fill_h))

            # Base Line (Green)
            pygame.draw.line(canvas, (0, 180, 0), (x - 5, px_base), (x + thickness + 5, px_base), 3)
            # Shifted left (x - 45)
            canvas.blit(self.hud_font.render("Base", True, (0, 180, 0)), (x - 45, px_base - 8))
            
            # Actual Line (Black)
            pygame.draw.line(canvas, (0, 0, 0), (x - 5, px_act), (x + thickness + 5, px_act), 4)
            # Shifted left (x - 35)
            canvas.blit(self.hud_font.render("Act", True, (0, 0, 0)), (x - 35, px_act - 8))
            
            # Labels (Shifted left to avoid right edge clipping)
            canvas.blit(self.hud_font.render(title, True, (0,0,0)), (x - 45, y - 40))
            canvas.blit(self.hud_font.render(pos_label, True, (0,0,0)), (x - 55, y - 20))
            canvas.blit(self.hud_font.render(neg_label, True, (0,0,0)), (x - 55, y + length + 5))

    def _render_frame(self, shap_values=None):
        self._pre_render()
        
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        
        width = self.unwrapped.window_width
        height = self.unwrapped.window_height
        screen_coords = self.unwrapped.screen_coords
        ac_idx = bs.traf.id2idx('KL001')
        
        # --- SHAP Data Extraction ---
        phi_hdg, phi_spd = np.zeros(10), np.zeros(10)
        max_hdg, max_spd = 1.0, 1.0
        base_hdg, base_spd = 0.0, 0.0
        act_hdg, act_spd = 0.0, 0.0
        
        if shap_values is not None and self.xrl_rendering:
            phi_hdg = shap_values.values[0][:, 0]  # Heading contributions
            phi_spd = shap_values.values[0][:, 1]  # Speed contributions
            max_hdg = max(0.01, np.max(np.abs(phi_hdg)))
            max_spd = max(0.01, np.max(np.abs(phi_spd)))
            base_hdg = float(shap_values.base_values[0][0])
            base_spd = float(shap_values.base_values[0][1])
            act_hdg = float(self.last_action[0])
            act_spd = float(self.last_action[1])

        # --- Draw Obstacles (Primary Saliency: Heading) ---
        for i, vertices in enumerate(self.unwrapped.obstacle_vertices):
            points = []
            obs_center_x, obs_center_y = 0, 0
            for coord in vertices:
                lat_ref, lon_ref = coord[0], coord[1]
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis * NM2KM
                x_ref = (np.sin(np.deg2rad(qdr)) * dis) / MAX_DISTANCE * width
                y_ref = (-np.cos(np.deg2rad(qdr)) * dis) / MAX_DISTANCE * width
                points.append((x_ref, y_ref))
                obs_center_x += x_ref
                obs_center_y += y_ref
            
            # Compute center for tethering
            obs_center_x /= len(vertices)
            obs_center_y /= len(vertices)
            
            # Color by Heading Influence
            poly_color = self._get_diverging_color(phi_hdg[i], max_hdg) if (shap_values is not None and self.xrl_rendering) else (50,50,50)
            pygame.draw.polygon(canvas, poly_color, points)
            pygame.draw.polygon(canvas, (0,0,0), points, width=2) # Outline

            # --- Secondary Saliency (Speed Tethering) ---
            if shap_values is not None and self.xrl_rendering:
                # If speed influence is greater than 15% of the max influence, draw a tether
                if abs(phi_spd[i]) > 0.15 * max_spd:
                    tether_color = (200, 0, 0) if phi_spd[i] > 0 else (0, 0, 200)
                    # Ownship coordinates mapping
                    qdr_ac, dis_ac = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                    ac_x = (np.sin(np.deg2rad(qdr_ac)) * dis_ac * NM2KM) / MAX_DISTANCE * width
                    ac_y = (-np.cos(np.deg2rad(qdr_ac)) * dis_ac * NM2KM) / MAX_DISTANCE * width
                    
                    pygame.draw.line(canvas, tether_color, (ac_x, ac_y), (obs_center_x, obs_center_y), width=2)

        # --- Draw Waypoint & Ownship ---
        qdr_ac, dis_ac = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        x_actor = (np.sin(np.deg2rad(qdr_ac)) * dis_ac * NM2KM) / MAX_DISTANCE * width
        y_actor = (-np.cos(np.deg2rad(qdr_ac)) * dis_ac * NM2KM) / MAX_DISTANCE * width
        
        # Waypoint
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], self.unwrapped.wpt_lat[0], self.unwrapped.wpt_lon[0])
        wpt_x = ((np.sin(np.deg2rad(wpt_qdr)) * wpt_dis * NM2KM) / MAX_DISTANCE) * width
        wpt_y = (-(np.cos(np.deg2rad(wpt_qdr)) * wpt_dis * NM2KM) / MAX_DISTANCE) * width
        pygame.draw.circle(canvas, (255, 255, 255), (wpt_x, wpt_y), radius=4)
        
        # Ownship
        pygame.draw.circle(canvas, (0, 0, 0), (x_actor, y_actor), radius=5)
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * 15
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * 15
        pygame.draw.line(canvas, (235, 52, 52), (x_actor, y_actor), (x_actor + heading_end_x, y_actor - heading_end_y), width=4)

        # --- XRL HUD & Bars ---
        if shap_values is not None and self.xrl_rendering:
            # HUD Overlay (Top Left)
            hud_bg = pygame.Surface((180, 50))
            hud_bg.set_alpha(200)
            hud_bg.fill((255, 255, 255))
            canvas.blit(hud_bg, (10, 10))
            canvas.blit(self.hud_font.render(f"Hdg Base: {base_hdg:+.2f}", True, (0,0,0)), (15, 15))
            canvas.blit(self.hud_font.render(f"Spd Base: {base_spd:+.2f}", True, (0,0,0)), (15, 35))

            # Left/Bottom Horizontal Bar (Heading)
            self._draw_manual_bar(
                canvas, x=60, y=height-40, length=200, thickness=15, 
                is_horizontal=True, base_val=base_hdg, actual_val=act_hdg,
                title="Heading Influence", pos_label="Right", neg_label="Left"
            )
            
            # Right Vertical Bar (Speed)
            self._draw_manual_bar(
                canvas, x=width-40, y=60, length=200, thickness=15, 
                is_horizontal=False, base_val=base_spd, actual_val=act_spd,
                title="Speed", pos_label="Accel", neg_label="Decel"
            )

        self._post_render(canvas)
