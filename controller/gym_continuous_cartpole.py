import math
from typing import Optional, Union

import numpy as np
import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class ContinuousCartPoleEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, masscart=1.0, masspole=0.1, length= 0.1, render_mode: Optional[str] = None):
        self.gravity = 9.81
        # self.masscart = 1.0
        # self.masspole = 0.1
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masscart + self.masspole
        # self.length = 0.5  # actually half the pole's length
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 100.0
        self.tau = 0.02 #0.02 0.01
        # self.kinematics_integrator = "euler"
        self.kinematics_integrator = "rk4"  # "euler" or "rk4"

        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.theta_threshold_radians = math.pi 
        # self.x_threshold = 2.4
        self.x_threshold = 5 # 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.force_mag], dtype=np.float32),
            high=np.array([self.force_mag], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 608 #600
        self.screen_height = 400
        self.state = None
        self.steps_beyond_terminated = None
        self.isopen = True

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        def dynamics(state, action):
            x, x_dot, theta, theta_dot = self.state

            force = float(action[0])
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            return np.array([x_dot, xacc, theta_dot, thetaacc], dtype=np.float32)
        
        y = self.state
        dt = self.tau

        if self.kinematics_integrator == "euler":
            x_current, x_dot_current, theta_current, theta_dot_current = y
            x_dot, xacc, theta_dot, thetaacc = dynamics(y, action)
            # x = x + self.tau * x_dot
            x = x_current + self.tau * x_dot_current
            # x_dot = x_dot + self.tau * xacc
            x_dot = x_dot_current + self.tau * xacc
            # theta = theta + self.tau * theta_dot
            theta = theta_current + self.tau * theta_dot_current
            # theta_dot = theta_dot + self.tau * thetaacc
            theta_dot = theta_dot_current + self.tau * thetaacc

        elif self.kinematics_integrator == "rk4":
            k1 = dynamics(y, action)
            k2 = dynamics(y + dt / 2 * k1, action)
            k3 = dynamics(y + dt / 2 * k2, action)
            k4 = dynamics(y + dt * k3, action)

            y_next = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x, x_dot, theta, theta_dot = y_next
        else:
            x_dot, xacc, theta_dot, thetaacc = dynamics(y, action)
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # terminated = (
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        #     or theta < -self.theta_threshold_radians
        #     or theta > self.theta_threshold_radians
        # )

        terminated = False

        reward = 1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}, action

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        # self.state = self.np_random.uniform(low=low, high=high, size=(4,))

        if options is not None and "init_state" in options:
            self.state = np.array(options["init_state"], dtype=np.float32)
        else:
            self.state = self.np_random.uniform(
                low=-0.05, high=0.05, size=(4,)
            ).astype(np.float32)

        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), {}

    # def render(self):
    #     # identical to original; use pygame or matplotlib if needed
    #     pass
    
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.__class__.__name__}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0
        carty = 100
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        # print(f"cartx: {cartx}, carty: {carty+ axleoffset}, polewidth/2: {polewidth / 2}")
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )



    def close(self):
        self.isopen = False
