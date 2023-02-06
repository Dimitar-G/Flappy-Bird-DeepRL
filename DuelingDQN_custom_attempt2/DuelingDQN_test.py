import time
import flappy_bird_gym
from deep_q_learning import DuelingDQN
from keras.layers import Dense


if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-v0")
    env.reset()
    env.render()

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model_layers = [
        Dense(32, activation='relu'),
        Dense(32, activation='relu')
    ]

    # num_episodes = 5001
    learning_rate = 0.01
    discount_factor = 0.99
    batch_size = 64
    memory_size = 2048
    # epsilon = 0.1

    agent = DuelingDQN(state_space_shape, num_actions, learning_rate,
                       discount_factor, batch_size, memory_size)

    agent.build_model(model_layers)

    agent.load('flappy', 400)

    total_reward = 0
    done = False
    state = env.reset()
    env.render()
    while not done:
        action = agent.get_action(state, 0)
        state, reward, done, info = env.step(action)
        print(f'{state}, {reward}, {info}, {done}')
        total_reward += reward
        env.render()
        time.sleep(1 / 50)

    print(f'Total reward: {total_reward}')
    env.close()
