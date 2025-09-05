import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PongEnvGym(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, width=400, height=300, paddle_width=10, paddle_height=60, ball_size=8):
        super(PongEnvGym, self).__init__()

        # Environment settings
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_size = ball_size

        # Gymnasium API: Observation and Action spaces
        # Observation: ball_x, ball_y, ball_vel_x, ball_vel_y, paddle_y
        obs_low = np.array([0, 0, -5, -5, 0], dtype=np.float32)
        obs_high = np.array([width, height, 5, 5, height - paddle_height], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Actions: -1 = up, 0 = stay, 1 = down
        self.action_space = spaces.Discrete(3)

        # Pygame init
        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Paddle position
        self.paddle_y = self.height // 2 - self.paddle_height // 2
        self.paddle_speed = 6

        # Ball position & velocity
        self.ball_x = float(self.ball_size // 2) #self.width // 2
        self.ball_y = self.height // 2
        self.ball_vel_x = np.random.choice([-3, 3])
        self.ball_vel_y = np.random.choice([-2, 2])

        obs = self._get_state()
        return obs, {}

    def step(self, action):
        # Action control
        if action == 0:   # Up
            self.paddle_y -= self.paddle_speed
        elif action == 2: # Down
            self.paddle_y += self.paddle_speed
        
        self.paddle_y = np.clip(self.paddle_y, 0, self.height - self.paddle_height)

        # Ball movement
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Ball bounce (top/bottom)
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_vel_y *= -1

        # Ball bounce on paddle
        if (self.ball_x >= self.width - self.paddle_width - self.ball_size and
            self.paddle_y < self.ball_y < self.paddle_y + self.paddle_height):
            self.ball_x = self.width - self.paddle_width - self.ball_size - 1  # prevent tunneling
            self.ball_vel_x *= -1

        # Check scoring
        terminated = False
        reward = 0
        if self.ball_x <= 0:
            reward = 1  # Agent scores
            terminated = True
        elif self.ball_x >= self.width:
            reward = -1 # Agent misses
            terminated = True

        obs = self._get_state()
        return obs, reward, terminated, False, {}

    def _get_state(self):
        return np.array([self.ball_x, self.ball_y, self.ball_vel_x, self.ball_vel_y, self.paddle_y], dtype=np.float32)

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Active Inference Pong")

        self.screen.fill((0, 0, 0))

        # Paddle
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.width - self.paddle_width, self.paddle_y, self.paddle_width, self.paddle_height))
        # Ball
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.ball_x, self.ball_y, self.ball_size, self.ball_size))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()


# Example usage
if __name__ == "__main__":
    env = PongEnvGym()
    obs, _ = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
