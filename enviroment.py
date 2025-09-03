import time
import os
from Actions import Actions
import numpy as np
import cv2


class FlappyBirdEnv:
    def __init__(self, config):
        self.actions = Actions()
        self.vision_model = self.YOLO('runs/detect/train4/weights/best.pt')
        self.image_path = os.path.join("game-images", "screenshot.png")

        self.game_over = False
        self.num_pipes = 0

        self.available_actions = self.get_available_actions()

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
            return self._getstate(), reward, self.game_over

        if action_index == 1:  # Flap action
            self.actions.press_space()

        self.game_over = False
        reward = self._compute_reward(self._getstate()) + self.num_pipes
        reward += 1  # Small reward for staying alive
        return self._getstate(), reward, self.game_over

    def _getstate(self):
        # Capture the current game state
        self.actions.capture_game(self.image_path)
        img = np.array(self.image_path)
        img_final = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.vision_model(img_final, conf=0.5)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # Tensor containing [x_min, y_min, x_max, y_max] for each box

        class_indices = results[0].boxes.cls.cpu().numpy()  # Class indices

        # Sort boxes based on class indices , so the fsrt box is always the bird
        paired = sorted(zip(boxes, class_indices), key=lambda x: x[1])

        sorted_boxes, sorted_class_indices = zip(*paired)

        # Convert the tensor to a list or NumPy array if needed
        return sorted_boxes

    def _compute_reward(self, state):
        if state is None:
            return 0
        bird_box = state[0]  # Assuming the first box is the bird
        closest_pipe = min(state[1:], key=lambda pipe: abs(pipe[0] - state[0][0]))
        if bird_box[1] < closest_pipe[1] and bird_box[3] > closest_pipe[3]:
            self.num_pipes += 2
            return 10  # Reward for passing through the pipe

    def get_available_actions(self):
        actions = [0, 1]  # 0: Do nothing, 1: Flap
        return actions

