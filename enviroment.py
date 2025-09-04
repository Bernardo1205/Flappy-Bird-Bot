import time
import os
from Actions import Actions
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from ultralytics import YOLO


class FlappyBirdEnv(gym.Env):
    def __init__(self, config):
        super(FlappyBirdEnv, self).__init__()
        self.actions = Actions()
        self.vision_model = YOLO('runs/detect/train4/weights/best.pt')
        self.image_path = os.path.join("game-images", "screenshot.png")

        self.action_space = spaces.Discrete(len(self.get_available_actions()))
        self.obs_shape = self._getstate().shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)

        self.MAX_PIPES = 8
        self.game_over = False
        self.num_pipes = 0

    def reset(self):
        time.sleep(3)
        self.actions.click_start()
        self.num_pipes = 0
        self.game_over = False
        return self._getstate()

    def step(self, action_index):

        if self.actions.detect_game_over():
            reward = self._compute_reward(self._getstate())
            reward -= 100  # Penalty for game over
            self.game_over = True
            info = {}
            terminated = truncated  = self.game_over
            return self._getstate(), reward, terminated, truncated , info

        if action_index == 1:  # Flap action
            self.actions.press_space()

        self.game_over = False
        reward = self._compute_reward(self._getstate()) + self.num_pipes
        reward += 1  # Small reward for staying alive
        info = {}
        terminated = truncated = self.game_over
        return self._getstate(), reward, terminated, truncated, info

    def _getstate(self):
        # Capture the current game state
        self.actions.capture_game(self.image_path)
        img= cv2.imread(self.image_path)
        img_final = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.vision_model(img_final, conf=0.5)

        boxes = results[0].boxes.xyxy.cpu().tolist()  # Tensor containing [x_min, y_min, x_max, y_max] for each box
        class_indices = results[0].boxes.cls.cpu().tolist()  # Class indices

        bird_box = np.array(boxes.pop(class_indices.index(0)))  # Assuming class index 0 is the bird
        pipes_boxes = np.array(boxes)  # Remaining boxes are pipes

        sorted_boxes = sorted(pipes_boxes, key=lambda pipe: abs(pipe[0] - bird_box[0]))
        sorted_boxes.insert(0, bird_box)  # Insert the bird box at the beginning

        #Pad or truncate the number of boxes so the state is always the same size
        if len(sorted_boxes) < self.MAX_PIPES:
            sorted_boxes += [np.array([0, 0, 0, 0])] * (self.MAX_PIPES - len(sorted_boxes))

        sorted_boxes_dist = sorted_boxes[:self.MAX_PIPES]

        #Flaten because we want a 1D array
        state = [coord for box in sorted_boxes_dist for coord in box]
        return np.array(state , dtype=np.float32)

    def _compute_reward(self, state):
        if state is None:
            return 0

        bird_box = state[:4]  # Assuming the first box is the bird
        pipe1 = state[4:8]
        pipe2 = state[8:12]

        upper_pipe = pipe1 if pipe1[1] < pipe2[1] else pipe2
        lower_pipe = pipe2 if pipe1[1] < pipe2[1] else pipe1

        # [x_min, y_min, x_max, y_max]

        if bird_box[0] > upper_pipe[0] and bird_box[2] < upper_pipe[2]  and bird_box[1] > upper_pipe[3] and bird_box[3] < lower_pipe[1]:
            self.num_pipes += 2
            return 10  # Reward for passing through the pipe
        return 0
    def get_available_actions(self):
        actions = [0, 1]  # 0: Do nothing, 1: Flap
        return actions
