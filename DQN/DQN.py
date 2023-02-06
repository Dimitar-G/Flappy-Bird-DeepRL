import time
from tqdm import tqdm
import flappy_bird_gym
from deep_q_learning import DQN
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.optimizers import Adam


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(Adam(learning_rate=learning_rate), loss=MeanSquaredError())
    return model


if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-v0")

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    num_episodes = 2500
    learning_rate = 0.01
    discount_factor = 0.99
    batch_size = 64
    memory_size = 2048
    epsilon = 0.1

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    agent = DQN(state_space_shape, num_actions, model, target_model,
                learning_rate, discount_factor, batch_size, memory_size)

    training_start = time.time()
    print(f'Training strarted at {training_start}')

    for episode in tqdm(range(num_episodes)):

        episode_start = time.time()
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.update_memory(state, action, reward, new_state, done)
            state = new_state
        agent.train()

        episode_end = time.time()

        if episode % 50 == 0 and episode != 0:
            agent.update_target_model()
            print(f'Episode {episode} finished - {episode_end - episode_start}s')

        if episode % 200 == 0 and episode != 0:
            agent.save('flappy_dqn', episode)

    agent.save('flappy_dqn', episode)

    training_end = time.time()
    print(f'Training ended at {training_end}')
    print(f'Total training time: {training_end - training_start}s')

    print()
    env.close()
