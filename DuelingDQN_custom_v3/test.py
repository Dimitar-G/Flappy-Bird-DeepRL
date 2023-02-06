from tqdm import tqdm
import flappy_bird_gym
from deep_q_learning import DuelingDQN
from keras.layers import Dense

MIN_EPISODE = 10
MAX_EPISODE = 3000
MODEL = "DuelingDQN"
NUM_TESTS = 25

if __name__ == '__main__':

    env = flappy_bird_gym.make("FlappyBird-v0")

    for episode in tqdm(range(MIN_EPISODE, MAX_EPISODE + 1, 10)):

        env.reset()

        state_space_shape = env.observation_space.shape[0]
        num_actions = env.action_space.n

        model_layers = [
            Dense(32, activation='relu'),
            Dense(32, activation='relu')
        ]

        # num_episodes = 5001
        learning_rate = 0.01
        discount_factor = 0.99
        batch_size = 32
        memory_size = 1024
        # epsilon = 0.1

        agent = DuelingDQN(state_space_shape, num_actions, learning_rate,
                           discount_factor, batch_size, memory_size)

        agent.build_model(model_layers)

        agent.load('flappy', episode)

        total_reward = 0

        for _ in range(NUM_TESTS):
            done = False
            state = env.reset()
            while not done:
                action = agent.get_action(state, 0)
                state, reward, done, info = env.step(action)
                total_reward += reward

        total_reward /= NUM_TESTS

        with open("test.txt", "a") as file:
            file.write(f"{MODEL} trained for {episode} episodes: Average reward in {NUM_TESTS} tests {total_reward}\n")
            file.flush()

        with open("test.csv", "a") as file:
            file.write(f"{episode},{total_reward}\n")
            file.flush()

    env.close()
