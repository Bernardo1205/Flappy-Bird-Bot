import pyautogui

class Actions:
    def __init__(self):
        window = pyautogui.getActiveWindow()
        self.window_top =window.top
        self.window_left = window.left
        self.window_width = window.width
        self.window_height = window.height

    def capture_game(self,save_path):
        # Capture the game window
        screenshot = pyautogui.screenshot(region=(self.window_left, self.window_top, self.window_width, self.window_height))
        screenshot.save(save_path)

    def press_space(self):
        pyautogui.press('space')

    def click_start(self):

        battel_buttom = (1397, 1124)

        if pyautogui.pixelMatchesColor(battel_buttom[0], battel_buttom[1], (22, 159, 73)):
            pyautogui.click(battel_buttom[0], battel_buttom[1])

