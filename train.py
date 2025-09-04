import time
from enviroment import FlappyBirdEnv as Enviroment
from stable_baselines3 import PPO

def train():
    time.sleep(5)  # Give you time to focus the game window

    env = Enviroment(config={})

    model = PPO('MlpPolicy', env, verbose=1 , learning_rate=1e-4 , batch_size=64)

    model.learn(total_timesteps=10000)

    model.save(path="model")

    model = PPO.load("model", env=env)

    episodes = 200

    for ep in range(episodes):
        state , info  = env.reset()
        print(f'Episode: {ep + 1}')
        done = False
        total_reward = 0

        while not done:
            action= model.predict(state, deterministic=True)
            print(action)
            next_state, reward, terminated , truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            if done:
                print(f'Episode: {ep + 1} / Reward: {total_reward}')
                # Environment will be reset in the next episode
                break

if __name__ == "__main__":
    train()