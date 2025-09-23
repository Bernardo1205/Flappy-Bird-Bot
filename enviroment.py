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

        self.image_path = os.path.join("game-images", 'screenshot.png')

        self.frame_skip = 3  # run YOLO every 3 frames
        self.frame_count = 0
        self.last_state = None  # cache last detection

        self.MAX_PIPES = 8
        self.game_over = False
        self.num_pipes = 0

        self.action_space = spaces.Discrete(len(self.get_available_actions()))
        self.obs_shape = self._getstate().shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)

    def reset(self, seed = None , options = None):
        super().reset(seed=seed)
        print("You Lost ")

        self.actions.click_start()
        time.sleep(0.2)
        self.actions.press_space()
        time.sleep(0.2)

        self.frame_count = 0
        self.last_state = None  # cache last detection
        self.num_pipes = 0
        self.game_over = False

        obs = self._getstate()
        return obs , {}

    def step(self, action_index):

        print("Step received action:", action_index)

        if self.actions.detect_game_over():
            reward = self._compute_reward(self._getstate())
            reward -= 100  # Penalty for game over
            self.game_over = True
            info = {}
            terminated = True
            truncated  = False
            return self._getstate(), reward, terminated, truncated , info

        if action_index == 1:  # Flap action
            print("Flap action executed")
            self.actions.press_space()
            time.sleep(0.05)  # Short delay to allow the game to register the action

        self.game_over = False
        reward = self._compute_reward(self._getstate()) + self.num_pipes
        reward += 1  # Small reward for staying alive
        info = {}
        terminated =False
        truncated = False
        return self._getstate(), reward, terminated, truncated, info

    def _getstate(self):

        self.frame_count += 1

        if self.frame_count % self.frame_skip != 0 and self.last_state is not None:
            return self.last_state

        # Capture the current game state
        self.actions.capture_game(self.image_path)
        img= cv2.imread(self.image_path)
        img_final = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.vision_model(img_final, conf=0.5)
        boxes = results[0].boxes.xyxy.cpu().tolist()  # Tensor containing [x_min, y_min, x_max, y_max] for each box
        class_indices = results[0].boxes.cls.cpu().tolist()  # Class indices

        #if the model falsely detects 2 birds we will remove the one that is farthest away from the center
        if len([x for x in class_indices if x == 0]) > 1:
            bird_candidates_index = [i for i, x in enumerate(class_indices) if x == 0]
            bird_candidates_boxes = [boxes[x] for x in bird_candidates_index]
            bird_box_index = boxes.index(min(bird_candidates_boxes, key=lambda b: abs(b[0] - img.shape[1] // 2)))

            # remove all the bird candidates except the one closest to the center
            [boxes.pop(x) for x in sorted(bird_candidates_index, reverse=True) if x != bird_box_index]
            [class_indices.pop(x) for x in sorted(bird_candidates_index, reverse=True) if x != bird_box_index]

        # if the model fails to detect the bird we will add a fake box at (0,0,0,0)
        if 0 not in class_indices:
            bird_box = np.array([0, 0, 0, 0])

        else:
            bird_box = np.array(boxes.pop(class_indices.index(0)))  # Assuming class index 0 is the bird

        pipes_boxes = np.array(boxes)  # Remaining boxes are pipes
        sorted_boxes = sorted(pipes_boxes, key=lambda pipe: abs(pipe[0] - bird_box[0]))
        sorted_boxes.insert(0, bird_box)  # Insert the bird box at the beginning

        #Pad or truncate the number of boxes so the state is always the same size
        if len(sorted_boxes) < self.MAX_PIPES:
            sorted_boxes += [np.array([0, 0, 0, 0])] * (self.MAX_PIPES - len(sorted_boxes))

        sorted_boxes_dist = sorted_boxes[:self.MAX_PIPES]

        #Flaten because we want a 1D array
        state = np.array([coord for box in sorted_boxes_dist for coord in box])
        self.last_state = state
        return state

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
  