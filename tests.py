import gym
import matplotlib.pyplot as plt
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers


def test_recent_memory():
    from rl.to_delete.memories import Recent
    recent = Recent(capacity=10)

    for i in range(recent.capacity):
        recent.append(i)

    print('recent memory:')
    print(recent.buffer)
    print(recent.retrieve(5))


def test_replay_memory():
    from rl.to_delete.memories import Replay
    replay = Replay(capacity=10)

    for i in range(replay.capacity):
        replay.append(i)

    print('replay memory:')
    print(replay.buffer)
    print(replay.retrieve(5))


def discount_cumsum(x, discount: float):
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def test_distribution():
    distribution = tfp.distributions.Categorical(probs=[0.7, 0.3])
    new_dist = tfp.distributions.Categorical(probs=[0.6, 0.4])
    print(distribution)

    e1 = distribution.sample(5)
    e2 = distribution.sample(5)

    print(e1, distribution.log_prob(e1), new_dist.log_prob(e1))
    print(e2, distribution.log_prob(e2))
    print(distribution.log_prob([0, 1]))
    print(distribution.log_prob([1, 0]))


def test_categorical(probs, action_shape=1):
    categorical = tfp.distributions.Categorical(probs=probs)
    print('categorical-1:', categorical.sample(1))
    print(f'categorical-{action_shape}:', categorical.sample(sample_shape=action_shape))


def test_independent_categorical(logits: list, action_shape=None):
    ind = tfp.distributions.Independent(
        distribution=tfp.distributions.Categorical(logits=logits),
        reinterpreted_batch_ndims=1)

    print(ind.sample())


def test_normal(mean):
    normal = tfp.distributions.Normal(loc=mean, scale=mean)
    print(normal.sample(5))


def test_beta(alpha, beta, num_samples=5):
    beta = tfp.distributions.Beta(alpha, beta)
    # samples in the range [0, 1]
    print(beta.sample(num_samples))


def test_categorical_vs_beta_prob(action_shape=1, num_samples=5):
    cat = tfp.distributions.Categorical(logits=[1.0, 2.0])
    beta = tfp.distributions.Beta(concentration0=1.0, concentration1=2.0)

    cat_actions = tf.reshape(tensor=cat.sample(num_samples), shape=(num_samples, 1))
    beta_actions = tf.reshape(tensor=beta.sample(num_samples), shape=(num_samples, 1))

    print(f'[Categorical] actions: {cat_actions}, log_prob: {cat.log_prob(cat_actions)}')
    print(f'[Beta] actions: {beta_actions}, log_prob: {beta.log_prob(beta_actions)}')


def test_dual_head_value_network(state_shape=1, units=1):
    # Network
    inputs = Input(shape=state_shape, dtype=tf.float32)
    x = Dense(units, activation='tanh')(inputs)
    x = Dense(units, activation='relu')(x)

    # Dual value head: intrinsic + extrinsic reward
    extrinsic_value = Dense(units=1, activation=None, name='extrinsic_head')(x)
    intrinsic_value = Dense(units=1, activation=None, name='intrinsic_head')(x)

    model = Model(inputs, outputs=[extrinsic_value, intrinsic_value])
    print('output_shape:', model.output_shape)
    print(model.summary())

    # Training (one gradient step)
    data = tf.constant([[1.0], [2.0], [3.0]])
    opt = optimizers.Adam()

    with tf.GradientTape() as tape:
        ext_values, int_values = model(data, training=True)
        print('values:')
        print(ext_values)
        print(int_values)

        ext_loss = -tf.square(ext_values)
        int_loss = -tf.square(int_values)
        losses = tf.reduce_mean(ext_loss * tf.constant(0.5) + int_loss * tf.constant(0.5))
        print('losses:')
        print(ext_loss)
        print(int_loss)
        print(losses)

    # grads = tape.gradient(tf.reduce_mean(losses), model.trainable_variables)
    grads = tape.gradient(losses, model.trainable_variables)
    print('gradients:\n', grads)
    info = opt.apply_gradients(zip(grads, model.trainable_variables))
    print('info:\n', info)


if __name__ == '__main__':
    # Memories:
    # test_recent_memory()
    # test_replay_memory()

    # GAE:
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=0.0)
    # test_generalized_advantage_estimation(gamma=0.99, lambda_=1.0)

    # Environments:
    # test_gym_env(num_episodes=200 * 5, max_timesteps=100, env='CartPole-v0')
    # print(discount_cumsum([1, 2, 3, 4], 1))

    # Distributions:
    # test_distribution()
    # test_categorical(action_shape=4, probs=[0.1, 0.3, 0.1, 0.5])
    # test_categorical(action_shape=1, probs=[0.1, 0.3, 0.1, 0.5])
    # test_independent_categorical(logits=[[1, 2], [3, 4]])
    # test_normal(mean=[1.0, 2.5])
    # test_beta(alpha=[1, 1], beta=2)
    # test_beta(alpha=1, beta=2, num_samples=1)
    # cat = tfp.distributions.Categorical(logits=[1, 2, 3, 4])
    # print(cat.log_prob([[1], [2], [3]]))
    test_categorical_vs_beta_prob(action_shape=1)

    # Networks:
    # test_dual_head_value_network()
    pass
