import time
import flappy_bird_gym
from deep_q_learning import DQN
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.optimizer_v2.adam import Adam


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(Adam(learning_rate=learning_rate), loss=MeanSquaredError())
    return model


if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-v0")
    env.reset()
    env.render()

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # num_episodes = 2500
    learning_rate = 0.01
    discount_factor = 0.99
    batch_size = 64
    memory_size = 2048
    # epsilon = 0.1

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    agent = DQN(state_space_shape, num_actions, model, target_model,
                learning_rate, discount_factor, batch_size, memory_size)

    agent.load('flappy_dqn', 2400)

    total_reward = 0
    done = False
    state = env.reset()
    env.render()
    while not done:
        action = agent.get_action(state, 0)
        state, reward, done, info = env.step(action)
        print(f'{state}, {reward}, {info}')
        total_reward += reward
        env.render()
        time.sleep(1 / 50)

    print(f'Total reward: {total_reward}')
    env.close()
