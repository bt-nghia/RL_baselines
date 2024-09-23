import gym
from dqn import Agent
import numpy as np


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=env.action_space.n,
        eps_end=0.01,
        input_dims=[8],
        lr = 0.003
    )

    scores, eps_history = [], []
    n_agmes = 500

    for _ in range(n_agmes):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            # print(env.step(action))
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode', _, 'score %.2f' %score,
              'avg_score %.2f' %avg_score,
              'epsilon %.2f' %agent.epsilon)