import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.horizontal_cr_env import ACTION_FREQUENCY,NUM_INTRUDERS,NM2KM,INTRUSION_DISTANCE,DISTANCE_MARGIN,AC_SPD
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn




# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class SaliencyHorizontalControl(gym.Wrapper):
    
    def __init__(self, env,safe_vals=None,debug=False):
        super().__init__(env)
        #self.unwrapped.window_size=(1024,1024)
        self.last_action = None  
        self.DEBUG = debug
        if safe_vals is not None:
            self.safe_vals = safe_vals
            
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        self.unwrapped.total_reward = 0
        self.unwrapped.total_intrusions = 0
        self.unwrapped.average_drift = np.array([])

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD)

        self.unwrapped._generate_conflicts()
        self.unwrapped._generate_waypoint()
        observation = self.unwrapped._get_obs()
        info = self.unwrapped._get_info()

        if self.unwrapped.render_mode == "human":
            self._render_frame()

        return observation, info
            
    def step(self, action, shap_values=None):
        
        self.unwrapped._get_action(action)
        self.last_action = action  # Store the last action

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self.unwrapped._get_obs()
                self._render_frame(shap_values=shap_values)

        observation = self.unwrapped._get_obs()
        reward, terminated = self.unwrapped._get_reward()

        info = self.unwrapped._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info
    
    def _render_frame(self,shap_values=None):
        if self.unwrapped.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.unwrapped.window = pygame.display.set_mode(self.unwrapped.window_size)

        if self.unwrapped.clock is None and self.render_mode == "human":
            self.unwrapped.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.window is not None:
                    pygame.display.quit()
                self.close()
                
        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.unwrapped.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.unwrapped.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2-heading_end_x/2,self.unwrapped.window_height/2+heading_end_y/2),
            ((self.unwrapped.window_width/2)+heading_end_x/2,(self.unwrapped.window_height/2)-heading_end_y/2),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.unwrapped.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.unwrapped.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2,self.unwrapped.window_height/2),
            ((self.unwrapped.window_width/2)+heading_end_x,(self.unwrapped.window_height/2)-heading_end_y),
            width = 1
        )

        if self.DEBUG:
            
            #plot one intrude (now plotted relative to ownship heading)
            color = (0,255,0)

            # compute relative bearing from cos/sin (these encode ac_hdg - qdr)
            rel_bearing_rad = np.arctan2(self.safe_vals["sin"], self.safe_vals["cos"])  # rel = ac_hdg - qdr
            rel_bearing_deg = np.rad2deg(rel_bearing_rad)
            # convert to global bearing from ownship to intruder
            int_qdr = (bs.traf.hdg[ac_idx] - rel_bearing_deg) % 360

            # position in km and then screen coords
            dist_km = self.safe_vals["dist"] * NM2KM
            x_pos = (self.unwrapped.window_width/2) + (np.sin(np.deg2rad(int_qdr)) * dist_km / max_distance) * self.unwrapped.window_width
            y_pos = (self.unwrapped.window_height/2) - (np.cos(np.deg2rad(int_qdr)) * dist_km / max_distance) * self.unwrapped.window_height

            # compute intruder heading: rotate local (dx,dy) by ownship heading
            heading_mag = np.sqrt(self.safe_vals["dx"]**2 + self.safe_vals["dy"]**2)
            if heading_mag > 1e-8:
                local_heading_rad = np.arctan2(self.safe_vals["dx"], self.safe_vals["dy"])  # matches sin/cos->angle convention used for drawing
                local_heading_deg = np.rad2deg(local_heading_rad)
                heading_global_deg = (bs.traf.hdg[ac_idx] + local_heading_deg) % 360

                heading_end_x = ((np.sin(np.deg2rad(heading_global_deg)) * ac_length)/max_distance)*self.unwrapped.window_width
                heading_end_y = ((np.cos(np.deg2rad(heading_global_deg)) * ac_length)/max_distance)*self.unwrapped.window_width

                pygame.draw.line(canvas,
                    color,
                    (x_pos,y_pos),
                    ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                    width = 4
                )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.unwrapped.window_width,
                width = 2
            )

        # draw intruders
        ac_length = 3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.unwrapped.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.unwrapped.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # # determine color
            # if int_dis < INTRUSION_DISTANCE:
            #     color = (220,20,60)
            # else: 
            #     color = (80,80,80)
            
            if shap_values is not None:
                if i < len(shap_values.values[0]):
                    saliency = shap_values.values[0][i]
                    if saliency > 0:
                        color = (min(255, int(255 * saliency)), 0, 0)
                    else:
                        color = (0, 0, min(255, int(-255 * saliency)))
                else:
                    color = (80,80,80)
            else:
                color = (80,80,80)
            

            x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.unwrapped.window_width
            y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.unwrapped.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.unwrapped.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.unwrapped.window_width

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.unwrapped.window_width,
                width = 2
            )

            # import code
            # code.interact(local=locals())

        # draw target waypoint
        for qdr, dis, reach in zip(self.unwrapped.wpt_qdr, self.unwrapped.waypoint_distance, self.unwrapped.wpt_reach):

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.unwrapped.window_width
            circle_y = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.unwrapped.window_width

            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                ((self.unwrapped.window_width/2)+circle_x,(self.unwrapped.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.unwrapped.window_width/2)+circle_x,(self.unwrapped.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.unwrapped.window_width,
                width = 2
            )

        
        
        # Draw legend for SHAP influence
        legend_x = 30
        legend_y = self.unwrapped.window_size[1] - 80
        legend_width = 200
        legend_height = 20
        font = pygame.font.SysFont(None, 24)

        # Draw sum of SHAP values above the legend
        if shap_values is not None:
            try:
                shap_sum = float(np.sum(shap_values.values))
            except Exception:
                shap_sum = float(np.sum(shap_values))
            sum_text = font.render(f"Sum of SHAP values: {shap_sum:.3f}", True, (0,0,0))
            canvas.blit(sum_text, (legend_x, legend_y - 30))
            baseline_text = font.render(f"Baseline: {shap_values.base_values[0][0]:.3f}", True, (0,0,0))
            canvas.blit(baseline_text, (legend_x, legend_y - 50))
            action_taken_text = font.render(f"Action taken: {self.last_action}", True, (0,0,0))
            text_rect = action_taken_text.get_rect()
            x = int(self.unwrapped.window_width / 2 - text_rect.width / 2)
            y = int(self.unwrapped.window_height / 2 - 30 - text_rect.height)
            canvas.blit(action_taken_text, (x, y))

        # Draw color scale: left (blue) to right (red)
        for i in range(legend_width):
            # Scale from -1 (left) to +1 (right)
            value = (i / legend_width) * 2 - 1
            if value < 0:
                color = (0, 0, min(255, int(-255 * value)))  # Blue for left
            else:
                color = (min(255, int(255 * value)), 0, 0)   # Red for right
            pygame.draw.line(canvas, color, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), 1)

        # Draw border
        pygame.draw.rect(canvas, (0,0,0), (legend_x, legend_y, legend_width, legend_height), 2)

        # Add text labels
        left_text = font.render('Left', True, (0,0,0))
        right_text = font.render('Right', True, (0,0,0))
        center_text = font.render('No Influence', True, (0,0,0))
        canvas.blit(left_text, (legend_x - 10, legend_y + legend_height + 5))
        canvas.blit(right_text, (legend_x + legend_width - 50, legend_y + legend_height + 5))
        canvas.blit(center_text, (legend_x + legend_width//2 - 50, legend_y + legend_height + 5))

        self.unwrapped.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.unwrapped.clock.tick(self.metadata["render_fps"])