import time
from tqdm import tqdm
import flappy_bird_gym
from deep_q_learning import DuelingDQN
from keras.layers import Dense


if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-v0")

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model_layers = [
        Dense(32, activation='relu'),
        Dense(32, activation='relu')
    ]

    num_episodes = 10001
    learning_rate = 0.001
    discount_factor = 0.99
    batch_size = 64
    memory_size = 2048
    epsilon = 0.1

    agent = DuelingDQN(state_space_shape, num_actions, learning_rate,
                       discount_factor, batch_size, memory_size)

    agent.build_model(model_layers)

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

        if episode % 25 == 0 and episode != 0:
            agent.update_target_model()
            print(f'Episode {episode} finished - {episode_end - episode_start}s')

        if episode % 200 == 0 and episode != 0:
            agent.save('flappy', episode)

    agent.save('flappy', episode)

    training_end = time.time()
    print(f'Training ended at {training_end}')
    print(f'Total training time: {training_end - training_start}s')

    print()
    env.close()
