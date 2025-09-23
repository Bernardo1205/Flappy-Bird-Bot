import pyautogui

class Actions:
    def __init__(self):
        window = pyautogui.size()
        self.window_top = 0
        self.window_left = 0
        self.window_width = window.width
        self.window_height = window.height
        self.battel_buttom = (1400, 1112)

    def capture_game(self,save_path):
        # Capture the game window
        screenshot = pyautogui.screenshot(region=(self.window_left, self.window_top, self.window_width, self.window_height))
        screenshot.save(save_path)

    def detect_game_over(self):
        if pyautogui.pixelMatchesColor(self.battel_buttom[0], self.battel_buttom[1], (22, 159, 73)):
            return True
        return False
    def press_space(self):
        pyautogui.press('space')

    def click_start(self):
        pyautogui.click(self.battel_buttom[0], self.battel_buttom[1])

