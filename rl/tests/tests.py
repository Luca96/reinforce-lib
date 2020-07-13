import random
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


def test_add_noise_action(n=4, state_shape=(2,), units=8, noise_std=0.05):
    # Network
    inputs = Input(shape=state_shape, dtype=tf.float32)
    x = Dense(units, activation='tanh')(inputs)
    action = Dense(units=1, activation='tanh')(x)
    model = Model(inputs, action)

    actions = model(tf.random.normal((n,) + state_shape))
    print('actions:', actions)
    noise = tf.random.normal(actions.shape, mean=0.0, stddev=noise_std)
    print('noise:', noise)
    print('action + noise:', actions + noise)


def test_noise_lambda_layer(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def get_model(noise=0.01):
        inp = Input(shape=(2,))
        x = Dense(2, activation='tanh')(inp)
        x = Lambda(lambda y: y + tf.random.normal(tf.shape(y), stddev=noise))(x)
        out = tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(x)
        return Model(inp, out)

    data = tf.random.normal((4, 2))
    r1 = get_model(noise=0.01)(data)
    r2 = get_model(noise=0.01)(data)
    for x, y in zip(r1, r2):
        print(x.numpy(), y.numpy())


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


def test_gym_spaces_change_shape(t=4):
    space = gym.spaces.Box(low=0, high=1, shape=(2,))
    print(space)

    # change shape
    space.shape = (t,) + space.shape
    print(space)


def test_gym_spaces_expand(t=4):
    def expand_space_shape(space: gym.Space):
        if isinstance(space, gym.spaces.Box):
            if utils.is_image(space):
                space.shape = space.shape[:2] + (space.shape[2] * t,)

            elif utils.is_vector(space):
                space.shape = (t,) + space.shape
            else:
                raise ValueError(f'Unsupported space type: {type(space)}!')

        elif isinstance(space, gym.spaces.Dict):
            for _, sub_space in space.spaces.items():
                expand_space_shape(space=sub_space)
        else:
            raise ValueError(f'Unsupported space type: {type(space)}!')

    example_space = gym.spaces.Dict(a=gym.spaces.Box(low=0, high=1, shape=(2, 3)),
                                    b=gym.spaces.Dict(c=gym.spaces.Box(low=0, high=1, shape=(5,)),
                                                      d=gym.spaces.Box(low=0, high=1, shape=(6, 7, 2))))
    print('before:', example_space)
    expand_space_shape(example_space)
    print('after:', example_space)


def test_tf_dataset_shard(size=20, num_shards=4, batch_size=5):
    dataset = tf.data.Dataset.range(size)

    if num_shards > 1:
        ds = dataset.shard(num_shards, index=0)

        for shard_index in range(1, num_shards):
            shard = dataset.shard(num_shards, index=shard_index)
            ds = ds.concatenate(shard)

        dataset = ds

    for batch in dataset.batch(batch_size):
        print(batch)


def test_data_to_batches(size=20, batch_size=5):
    data = tf.data.Dataset.range(size).as_numpy_iterator()

    for batch in utils.data_to_batches(list(data), batch_size):
        print(batch)


def test_predict_different_batch():
    def get_model(shape=(2,), batch=2):
        inp = Input(shape=shape, batch_size=batch)
        x = Dense(2)(inp)
        out = Dense(4, activation='softmax')(x)
        return Model(inp, out)

    model = get_model()
    data = tf.random.normal((1, 2))
    # model.predict(x=data, batch_size=1)
    model(data)
    # model(data, training=True)
    pass


def test_iterate_layer_output():
    def get_model():
        inp = Input(shape=(2,))
        x = Dense(4, name='layer')(inp)
        return Model(inp, Dense(1)(x))

    model = get_model()
    model(tf.random.normal((1, 2)))
    layer: Layer = model.get_layer(name='layer')

    weights, bias = layer.weights
    print('weights:')
    for w in weights.value():
        print(w)

    print('bias:')
    for b in bias.value():
        print(b)


def test_incremental_mean(x: list, alpha=1.0):
    m = [x[0]]

    for i in range(1, len(x)):
        c1 = (i - 1) / i
        c2 = 1 / i

        m_i = (c1 * m[i - 1]) + (c2 * alpha * x[i])
        m.append(m_i)

    mean = np.mean(x)
    print(f'mean: {mean} vs i.mean: {np.round(m, 2)}')


def test_incremental_mean2(x: list):
    # Source: http://datagenetics.com/blog/november22017/index.html
    m = [x[0]]

    for i in range(1, len(x)):
        m_i = m[i - 1] + (x[i] - m[i - 1]) / i
        m.append(m_i)

    mean = np.mean(x)
    print(f'mean: {mean} vs i.mean: {np.round(m, 2)}')


def test_incremental_std(x: list):
    # Source: http://datagenetics.com/blog/november22017/index.html
    m = [0]
    s = [0]

    for i in range(1, len(x) + 1):
        m_i = m[i - 1] + (x[i - 1] - m[i - 1]) / i
        s_i = s[i - 1] + (x[i - 1] - m[i - 1]) * (x[i - 1] - m_i)

        m.append(m_i)
        s.append(s_i)

    std = np.std(x)
    inc_std = np.sqrt(s[-1] / len(x))
    print(f'std: {std} vs i.std: {inc_std}')


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
    # test_add_noise_action()
    # test_noise_lambda_layer()
    # test_predict_different_batch()
    # test_iterate_layer_output()

    # Data:
    # test_tf_data_api()
    # test_space_to_flat_spec()
    # test_gym_spaces_change_shape()
    # test_gym_spaces_expand()
    # test_tf_dataset_shard()
    # test_data_to_batches()

    # test_incremental_mean([1, 2, 3, 3, 0, -1, -6, 13, 2, -7], alpha=0.9)
    # test_incremental_mean([1, 2, 3, 3, 0, -1, -6, 13, 2, -7])
    # test_incremental_mean2([1, 2, 3, 3, 0, -1, -6, 13, 2, -7])
    # test_incremental_std([1, 2, 3, 3, 0, -1, -6, 13, 2, -7])
    pass
