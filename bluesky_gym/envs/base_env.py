
import gymnasium as gym
import gymnasium as gym
import pygame
import os
import imageio
import bluesky as bs
import numpy as np



class BaseEnv(gym.Env):
    def __init__(self,export_gifs_path=None,fps=5):
        """This is a base class for gym envs. This has common functinality like pre_render function, saving a frame, exporting a gif

        Args:
            env (gym.evn): env instance
            export_gifs_path (str, optional): String to export path, creates directories. Defaults to None.
            fps (int, optional): fps of the gif. Defaults to 5.
            xrl_rendering (bool, optional): Turn on or off the XRL specific rendering. Defaults to True.
        """ 
        self.export_gifs_path = export_gifs_path
        self._init_gif_folders()
        self.fps = fps
        self.frame_saved = False

        
    def _init_gif_folders(self):
        if self.export_gifs_path is not None:
            os.makedirs(self.export_gifs_path, exist_ok=True)
        # inside create two folder: frames and gifs
        if self.export_gifs_path is not None:
            self.frames_path = os.path.join(self.export_gifs_path, "frames")
            self.gifs_path = os.path.join(self.export_gifs_path, "gifs")
            os.makedirs(self.frames_path, exist_ok=True)
            os.makedirs(self.gifs_path, exist_ok=True)
            
    def export_episode_gif(self):
        if self.export_gifs_path is not None:
                # export gif from saved frames
                gif_filename = os.path.join(self.gifs_path, f"episode_{self.episode_counter}.gif")
                images = [imageio.imread(os.path.join(self.episode_frames_path, f"frame_{step}.png")) for step in range(1, self.step_counter + 1)]
                imageio.mimsave(gif_filename, images, fps=self.fps)

                
    def _post_render(self,canvas):
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        if self.export_gifs_path is not None and not self.frame_saved and self.episode_counter >=0:
            self.frame_saved = True
            # save frame to episode frames folder use the current step count as filename
            frame_filename = os.path.join(self.episode_frames_path, f"frame_{self.step_counter}.png")
            try:
                pygame.image.save(canvas, frame_filename)
            except pygame.error as e:
                print(f"Error saving frame {self.step_counter} of episode {self.episode_counter}: {e}")
    
    def _pre_render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.window is not None:
                    pygame.display.quit()
                self.close()
                exit()