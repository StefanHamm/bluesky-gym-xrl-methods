import gymnasium as gym
import numpy as np
import pygame
import os
import bluesky as bs
from bluesky_gym.envs.merge_env import ACTION_FREQUENCY, DISTANCE_MARGIN, INTRUSION_DISTANCE, HEADING_LENGTH_IN_SECONDS, NM2KM, RWY_LAT, RWY_LON, NUM_AC
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper

class SaliencyMergeControl(SaliencyMapV1Wrapper):
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
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame(shap_values=shap_values)

        observation = self.unwrapped._get_obs()
        reward, terminated = self.unwrapped._get_reward()
        info = self.unwrapped._get_info()
        
        if terminated and self.export_gifs_path is not None:
            self.export_episode_gif()

        return observation, reward, terminated, False, info

    def _get_diverging_color(self, phi, max_phi, default=(255, 255, 255), is_entity=False):
        """Red for >0, Blue for <0. Uses default if phi is 0."""
        if abs(phi) < 0.01:
            return (80, 80, 80) if is_entity else default
        norm = min(1.0, abs(phi) / (max_phi + 1e-6))
        return (255, int(255*(1-norm)), int(255*(1-norm))) if phi > 0 else (int(255*(1-norm)), int(255*(1-norm)), 255)

    def _draw_manual_bar(self, canvas, x, y, length, thickness, is_horizontal, base_val, actual_val, title, pos_label, neg_label):
        # Background Gradient
        for i in range(length):
            ratio = i / length
            color = (int(255*(1-ratio)), 0, int(255*ratio))
            if is_horizontal:
                pygame.draw.line(canvas, color, (x + i, y), (x + i, y + thickness))
            else:
                pygame.draw.line(canvas, color, (x, y + i), (x + thickness, y + i))
        pygame.draw.rect(canvas, (0,0,0), (x, y, length if is_horizontal else thickness, thickness if is_horizontal else length), 2)

        b_val, a_val = max(-1.0, min(1.0, base_val)), max(-1.0, min(1.0, actual_val))

        if is_horizontal:
            px_base = x + int(((b_val + 1.0) / 2.0) * length)
            px_act = x + int(((a_val + 1.0) / 2.0) * length)
            fill_x, fill_w = min(px_base, px_act), abs(px_act - px_base)
            if fill_w > 0:
                pygame.draw.rect(canvas, (200, 0, 0) if a_val > b_val else (0, 0, 200), (fill_x, y, fill_w, thickness))
            
            pygame.draw.line(canvas, (0, 180, 0), (px_base, y - 5), (px_base, y + thickness + 5), 3)
            canvas.blit(self.hud_font.render("Base", True, (0, 180, 0)), (px_base - 15, y - 20))
            pygame.draw.line(canvas, (0, 0, 0), (px_act, y - 5), (px_act, y + thickness + 5), 4)
            canvas.blit(self.hud_font.render("Act", True, (0, 0, 0)), (px_act - 10, y + thickness + 8))
            
            canvas.blit(self.hud_font.render(title, True, (0,0,0)), (x, y - 40))
            canvas.blit(self.hud_font.render(neg_label, True, (0,0,0)), (x - 40, y + 2))
            canvas.blit(self.hud_font.render(pos_label, True, (0,0,0)), (x + length + 10, y + 2))
        else:
            px_base = y + int(((1.0 - b_val) / 2.0) * length)
            px_act = y + int(((1.0 - a_val) / 2.0) * length)
            fill_y, fill_h = min(px_base, px_act), abs(px_act - px_base)
            if fill_h > 0:
                pygame.draw.rect(canvas, (200, 0, 0) if a_val > b_val else (0, 0, 200), (x, fill_y, thickness, fill_h))
            
            pygame.draw.line(canvas, (0, 180, 0), (x - 5, px_base), (x + thickness + 5, px_base), 3)
            canvas.blit(self.hud_font.render("Base", True, (0, 180, 0)), (x - 45, px_base - 8))
            pygame.draw.line(canvas, (0, 0, 0), (x - 5, px_act), (x + thickness + 5, px_act), 4)
            canvas.blit(self.hud_font.render("Act", True, (0, 0, 0)), (x - 35, px_act - 8))
            
            canvas.blit(self.hud_font.render(title, True, (0,0,0)), (x - 45, y - 40))
            canvas.blit(self.hud_font.render(pos_label, True, (0,0,0)), (x - 55, y - 20))
            canvas.blit(self.hud_font.render(neg_label, True, (0,0,0)), (x - 55, y + length + 5))

    def _render_frame(self, shap_values=None):
        self._pre_render()
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235)) 
        
        w, h = self.unwrapped.window_width, self.unwrapped.window_height
        cx, cy = w / 2, h / 2
        max_dist = 500
        
        # --- SHAP Data Extraction ---
        phi_hdg, phi_spd = np.zeros(8), np.zeros(8)
        max_hdg, max_spd = 1.0, 1.0
        b_hdg, b_spd, a_hdg, a_spd = 0.0, 0.0, 0.0, 0.0
        
        if shap_values is not None and self.xrl_rendering:
            phi_hdg = shap_values.values[0][:, 0]
            phi_spd = shap_values.values[0][:, 1]
            max_hdg, max_spd = max(0.01, np.max(np.abs(phi_hdg))), max(0.01, np.max(np.abs(phi_spd)))
            b_hdg, b_spd = float(shap_values.base_values[0][0]), float(shap_values.base_values[0][1])
            a_hdg, a_spd = float(self.last_action[0]), float(self.last_action[1])

        # --- Base Geometry Mapping ---
        # 1. Target FAF (Coalition 6 - Distance influence on Speed)
        c_faf = self._get_diverging_color(phi_spd[6], max_spd) if self.xrl_rendering else (255,255,255)
        pygame.draw.circle(canvas, c_faf, (cx, cy), radius=4, width=0)
        pygame.draw.circle(canvas, c_faf, (cx, cy), radius=(DISTANCE_MARGIN/max_dist)*w, width=2)

        # 2. Runway Target Line
        rwy_qdr, rwy_dis = bs.tools.geo.kwikqdrdist(self.unwrapped.wpt_lat, self.unwrapped.wpt_lon, RWY_LAT, RWY_LON)
        rwy_x = cx + (np.cos(np.deg2rad(rwy_qdr))*(rwy_dis * NM2KM)/max_dist)*w
        rwy_y = cy - (np.sin(np.deg2rad(rwy_qdr))*(rwy_dis * NM2KM)/max_dist)*h
        r_end_x = ((np.cos(np.deg2rad(180)) * 5000)/max_dist)*w
        r_end_y = ((np.sin(np.deg2rad(180)) * 5000)/max_dist)*w
        pygame.draw.line(canvas, (255,255,255), (rwy_x, rwy_y), (cx + r_end_x/2, cy - r_end_y/2), width=4)

        # 3. Ownship (Coalition 5 & 7 - Drift Hdg line, Airspeed Spd body)
        ac_idx = bs.traf.id2idx('KL001')
        own_qdr, own_dis = bs.tools.geo.kwikqdrdist(self.unwrapped.wpt_lat, self.unwrapped.wpt_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        own_x = cx + (np.cos(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_dist)*w
        own_y = cy - (np.sin(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_dist)*h
        
        hdg_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * 8)/max_dist)*w
        hdg_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * 8)/max_dist)*w
        
        c_own_body = self._get_diverging_color(phi_spd[7], max_spd, default=(0,0,0)) if self.xrl_rendering else (0,0,0)
        pygame.draw.line(canvas, c_own_body, (own_x, own_y), (own_x + hdg_end_x/2, own_y - hdg_end_y/2), width=5)

        # Route Target Line (Aircraft to FAF) colored by Drift
        c_drift_line = self._get_diverging_color(phi_hdg[5], max_hdg) if self.xrl_rendering else (255,255,255)
        pygame.draw.line(canvas, c_drift_line, (own_x, own_y), (cx, cy), width=1)

        # 4. Intruders (Coalitions 0-4)
        distances = bs.tools.geo.kwikdist_matrix(bs.traf.lat[0], bs.traf.lon[0], bs.traf.lat[1:], bs.traf.lon[1:])
        tracked_indices = np.argsort(distances)[:5] + 1 # Top 5 tracked IDs

        for i in range(1, NUM_AC):
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(self.unwrapped.wpt_lat, self.unwrapped.wpt_lon, bs.traf.lat[i], bs.traf.lon[i])
            ix = cx + (np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_dist)*w
            iy = cy - (np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_dist)*h
            
            i_hdg_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[i])) * 3)/max_dist)*w
            i_hdg_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[i])) * 3)/max_dist)*w

            color = (80, 80, 80) # Default Untracked Gray
            if i in tracked_indices and shap_values is not None and self.xrl_rendering:
                shap_idx = np.where(tracked_indices == i)[0][0]
                color = self._get_diverging_color(phi_hdg[shap_idx], max_hdg, is_entity=True)
                
                # Secondary Tether for Speed Influence
                if abs(phi_spd[shap_idx]) > 0.15 * max_spd:
                    t_color = (200, 0, 0) if phi_spd[shap_idx] > 0 else (0, 0, 200)
                    pygame.draw.line(canvas, t_color, (own_x, own_y), (ix, iy), width=2)
            elif not self.xrl_rendering and int_dis < INTRUSION_DISTANCE:
                color = (220, 20, 60) # Base Envs Warning Red

            pygame.draw.line(canvas, color, (ix, iy), (ix + i_hdg_end_x, iy - i_hdg_end_y), width=4)
            pygame.draw.circle(canvas, color, (ix, iy), radius=(INTRUSION_DISTANCE*NM2KM/max_dist)*w, width=2)

        # --- XRL HUD & Bars ---
        if shap_values is not None and self.xrl_rendering:
            labels = ["Int-1", "Int-2", "Int-3", "Int-4", "Int-5", "Drift", "Distance", "Airspeed"]
            top_hdg_idx, top_spd_idx = np.argmax(np.abs(phi_hdg)), np.argmax(np.abs(phi_spd))
            
            # HUD Overlay
            hud_bg = pygame.Surface((300, 50))
            hud_bg.set_alpha(200)
            hud_bg.fill((255, 255, 255))
            canvas.blit(hud_bg, (10, 10))
            canvas.blit(self.hud_font.render(f"Top Hdg: {labels[top_hdg_idx]} ({phi_hdg[top_hdg_idx]:+.2f})", True, (0,0,0)), (15, 15))
            canvas.blit(self.hud_font.render(f"Top Spd: {labels[top_spd_idx]} ({phi_spd[top_spd_idx]:+.2f})", True, (0,0,0)), (15, 35))

            # Left/Bottom Horizontal Bar (Heading)
            self._draw_manual_bar(canvas, x=60, y=h-40, length=200, thickness=15, is_horizontal=True, 
                                  base_val=b_hdg, actual_val=a_hdg, title="Heading", pos_label="Right", neg_label="Left")
            
            # Right Vertical Bar (Speed)
            self._draw_manual_bar(canvas, x=w-40, y=60, length=200, thickness=15, is_horizontal=False, 
                                  base_val=b_spd, actual_val=a_spd, title="Speed", pos_label="Accel", neg_label="Decel")

        self._post_render(canvas)
