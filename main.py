import rl
import numpy as np
import gym
import tensorflow_probability as tfp


from rl.agents import Agent


def gaussian_policy(action_shape):
    pass


def main():
    n = tfp.distributions.Normal(loc=[1.0, -1.0], scale=[1.0, 1.0])
    print(n.sample(1, seed=42).numpy())
    # print(n.log_prob(v).numpy())
    # print(n.prob(v).numpy())

    mvn = tfp.distributions.MultivariateNormalDiag(loc=[1., -1], scale_diag=[1.0, 1.0])
    print(mvn.sample(1, seed=42).numpy())


def gym_test():
    env = gym.make('CartPole-v0')
    agent = Agent(state_shape=env.observation_space.shape, action_shape=(1,), batch_size=1)

    for i_episode in range(20):
        observation = env.reset()

        for t in range(100):
            env.render()
            print(observation)

            observation = np.expand_dims(observation, axis=0)
            action = agent.act(observation)[0][0]
            action = action.numpy().astype(np.int)
            print(action)

            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()


if __name__ == '__main__':
    # main()
    gym_test()
