from tqdm import tqdm
import flappy_bird_gym
from deep_q_learning import DQN
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

MIN_EPISODE = 10
MAX_EPISODE = 610
MODEL = "DQN"
NUM_TESTS = 25


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(Adam(learning_rate=learning_rate), loss=MeanSquaredError())
    return model


if __name__ == '__main__':

    env = flappy_bird_gym.make("FlappyBird-v0")

    for episode in tqdm(range(MIN_EPISODE, MAX_EPISODE + 1, 10)):

        env.reset()

        state_space_shape = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # num_episodes = 5001
        learning_rate = 0.01
        discount_factor = 0.99
        batch_size = 64
        memory_size = 2048
        # epsilon = 0.1

        model = build_model(state_space_shape, num_actions, learning_rate)
        target_model = build_model(state_space_shape, num_actions, learning_rate)

        agent = DQN(state_space_shape, num_actions, model, target_model,
                    learning_rate, discount_factor, batch_size, memory_size)

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
