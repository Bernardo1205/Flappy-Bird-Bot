import time
from enviroment import FlappyBirdEnv as Enviroment
from stable_baselines3 import PPO
from pynput import keyboard


class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nShutdown requested - cleaning up...")
                self.should_exit = True
        except AttributeError:
            pass  # Special key pressed

    def is_exit_requested(self):
        return self.should_exit

def train():
    time.sleep(3)  # Give you time to focus the game window

    env = Enviroment(config={})

    model = PPO('MlpPolicy', env, verbose=1 , learning_rate=1e-4 , batch_size=64)

    model.learn(total_timesteps=10000)

    model.save(path="model")

    controler = KeyboardController()

    episodes = 200

    for ep in range(episodes):
        if controler.is_exit_requested():
            print('Exit required ')
            break
        state , info  = env.reset()
        print(f'Episode: {ep + 1}')
        done = False
        total_reward = 0

        while not done:

            action , _ = model.predict(state)
            next_state, reward, terminated , truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            if done:
                print(f'Episode: {ep + 1} / Reward: {total_reward}')
                # Environment     reset in the next episode
                break

if __name__ == "__main__":
    train()