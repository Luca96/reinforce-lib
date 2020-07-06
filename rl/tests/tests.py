import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from rl import utils


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


def test_tf_data_api(data_size=10, batch_size=4):
    data = [[i] for i in range(data_size)]
    print('data:', data)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    print('batched:', list(dataset.as_numpy_iterator()))
    print('batch-shuffled:', list(dataset.shuffle(buffer_size=batch_size).as_numpy_iterator()))


def test_dict_inputs():
    # Build model
    a = Input(shape=1, name='a')
    b = Input(shape=1, name='b')
    x = concatenate([a, b])
    x = Dense(16, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inputs=[a, b], outputs=x)
    model.summary()

    # Feed input
    result = model(dict(a=np.random.random((2, 1)),
                        b=np.random.random((2, 1))))
    print(result)


def test_space_to_flat_spec():
    space = gym.spaces.Dict(a=gym.spaces.Dict(b=gym.spaces.Discrete(2)),
                            c=gym.spaces.Box(low=0, high=1, shape=(2, 2)))
    print(utils.space_to_flat_spec(space, name='space'))


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
    # test_categorical_vs_beta_prob(action_shape=1)

    # Networks:
    # test_dual_head_value_network()
    # test_dict_inputs()

    # Data:
    # test_tf_data_api()
    test_space_to_flat_spec()
    pass
