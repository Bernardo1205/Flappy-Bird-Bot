"""
    This enviroment in a new update 2.0 version of the Flappy Bird enviroment in whch it uses
    A new state representation t try to improve the performance of the RL agent.

    Changes :
        The state will consist now only of the bird box and the next gap that the bird must pass through. This is
        because the agent only needs to focus on the immediate challenge to make decisions, and parsing all the gaps would
        add unnecessary complexity and computational overhead.  By focusing on the next space_throught, you simplify the
        state representation, reduce the dimensionality of the input, and ensure the agent is not distracted by irrelevant
        information. This approach aligns with the principle of providing the agent with only the most relevant information
        for decision-making.

        Resize the image before passing it to the YOLO model to improve detection speed.

        Change in the reward function to give a reward base on the gap and include a small treshold to consider the bird
        
"""

import time
import os
import pyautogui
from Actions import Actions
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from ultralytics import YOLO


class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.actions = Actions()
        self.vision_model = YOLO('runs/detect/train4/weights/best.pt')

        self.image_path = os.path.join("game-images", 'screenshot.png')

        self.frame_skip = 3  # run YOLO every 3 frames
        self.frame_count = 0
        self.last_state = None  # cache last detection

        self.treshold = 2  # Margin of error for being inside the gap
        self.MAX_PIPES = 3
        self.game_over = False
        self.num_pipes = 0

        self.action_space: spaces.Discrete = spaces.Discrete(len(self.get_available_actions()))

        self.obs_shape = self._getstate().shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
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
        return obs, {}

    def step(self, action_index):

        print("action:", action_index)

        if self.actions.detect_game_over():
            reward = self._compute_reward(self._getstate())
            reward -= 100  # Penalty for game over
            self.game_over = True
            info = {}
            terminated = True
            truncated = False
            return self._getstate(), reward, terminated, truncated, info

        if action_index == 1:  # Flap action
            self.actions.press_space()
            time.sleep(0.05)  # Short delay to allow the game to register the action

        self.game_over = False
        reward = self._compute_reward(self._getstate()) + self.num_pipes
        reward += 1  # Small reward for staying alive
        info = {}
        terminated = False
        truncated = False
        return self._getstate(), reward, terminated, truncated, info

    def _getstate(self):

        self.frame_count += 1

        if self.frame_count % self.frame_skip != 0 and self.last_state is not None:
            return self.last_state

        # Capture the current game state
        self.actions.capture_game(self.image_path)
        img = cv2.imread(self.image_path)
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

        # New : Ensure the first two boxes are the upper and lower pipes of the next gap
        if len(sorted_boxes) >= 2:
            upper_pipe, lower_pipe = sorted(sorted_boxes[:2], key=lambda pipe: pipe[1])  # Sort by y_min
            sorted_boxes[:2] = [upper_pipe, lower_pipe]


        else:  # If no pipes detected, create a fake gap in the center
            img_h, img_w = img_final.shape[:2]
            # Place a fake gap roughly at the middle height and a bit to the right (e.g., 70% width)
            fake_gap_width = img_w * 0.15
            gap_x_min = img_w * 0.6
            gap_x_max = gap_x_min + fake_gap_width

            gap_y_center = img_h / 2
            gap_height = img_h * 0.25  # adjustable depending on your game
            gap_y_min = gap_y_center - gap_height / 2
            gap_y_max = gap_y_center + gap_height / 2

            upper_pipe = np.array([gap_x_min, 0, gap_x_max, gap_y_min])
            lower_pipe = np.array([gap_x_min, gap_y_max, gap_x_max, img_h])

            sorted_boxes = [upper_pipe, lower_pipe]

        sorted_boxes.insert(0, bird_box)  # Insert the bird box at the beginning

        # New : Instead of parsing the bird box and then the pipes boxes separately, we will parse the model the bird
        # box and the gap box between the pipes which is the space the bird has to fly through

        # [x_min, y_min, x_max, y_max]
        space_throught = np.array([
            sorted_boxes[1][0],  # x_min of upper pipe
            sorted_boxes[1][3],  # y_max of upper pipe
            sorted_boxes[2][2],  # x_max of lower pipe
            sorted_boxes[2][1]  # y_min of lower pipe
        ])

        sorted_boxes_dist = [sorted_boxes[0], space_throught]

        # Flaten because we want a 1D array
        state = np.array([coord for box in sorted_boxes_dist for coord in box])
        self.last_state = state

        # Then normalize (assuming 2 boxes → 8 coords)
        img_h, img_w = img_final.shape[:2]
        state = state / np.array([img_w, img_h, img_w, img_h] * (len(state) // 4))
        return state

    def _compute_reward(self, state):
        if state is None:
            return 0

        bird_box = state[:4]
        gap = state[4:8]

        bird_center_y = (bird_box[1] + bird_box[3]) / 2
        gap_center_y = (gap[1] + gap[3]) / 2

        # reward based on how close the bird is to the center of the gap
        distance = abs(bird_center_y - gap_center_y)
        reward = -distance / 100  # normalize distance penalty

        # bonus for survival
        reward += 1.0

        in_gap_horizontally = (
                bird_box[2] >= gap[0] - self.treshold and
                bird_box[0] <= gap[2] + self.treshold
        )
        in_gap_vertically = (
                bird_box[1] >= gap[1] and
                bird_box[3] <= gap[3]
        )

        # Bonus if the bird is inside the gap horizontally (within threshold) and vertically
        if in_gap_horizontally and in_gap_vertically:
            reward += 10.0  # large positive reward for being inside gap

        return reward

    def get_available_actions(self):
        actions = [0, 1]  # 0: Do nothing, 1: Flap
        return actions

    def isOpenGame(self):
        battel_buttom = (1400, 1112)
        if pyautogui.pixelMatchesColor(battel_buttom[0], battel_buttom[1],
                                       (23, 166, 76)) or pyautogui.pixelMatchesColor(battel_buttom[0], battel_buttom[1],
                                                                                     (22, 159, 73)):
            return True
        print(pyautogui.pixel(battel_buttom[0], battel_buttom[1]))
        return False
