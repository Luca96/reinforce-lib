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


def test_data_to_batches(size=20, batch_size=5, **kwargs):
    data = tf.data.Dataset.range(size).as_numpy_iterator()

    for batch in utils.data_to_batches(list(data), batch_size, **kwargs):
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


def test_dataset_sharding_scenarios():
    print('ideal')
    test_data_to_batches(size=50, batch_size=10, num_shards=5)  # (batch_size * num_shards) % size == 0
    print('-------')
    test_data_to_batches(size=100, batch_size=10, num_shards=5)
    print('-------')
    print(50 / (10 * 5))
    print(100 / (10 * 5))
    print(95 / (10 * 5))


def test_model_independent_distribution():
    import tensorflow_probability as tfp

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    def get_models():
        inp = Input(shape=(2,))
        x = Dense(2)(inp)

        out1 = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(x)

        out2 = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Independent(
                distribution=tfp.distributions.Categorical(logits=t)))(x)

        return Model(inp, out1), Model(inp, out2)

    model1, model2 = get_models()

    for data in [tf.random.normal((1, 2)), tf.random.normal((5, 2))]:
        print([x.numpy() for x in model1(data)])
        print([x.numpy() for x in model2(data)])
        print()


def test_rnn_sequences():
    def get_model():
        inp = Input(shape=(None, 2))
        gru = GRU(8, return_sequences=True, return_state=True)  # x == state if return_sequences=False
        x, state = gru(inp)
        return Model(inp, [x, state]), gru

    model, gru_layer = get_model()
    model.summary()

    data = tf.random.normal((1, 6, 2))
    y, h = model(data)
    print(y, h)

    # print(gru_layer.get_initial_state(h))  # prints [0, 0, ..., 0]
    # gru_layer.reset_states(states=h)  # CANNOT DO THIS! needs stateful=True)
    # print(model(data))


def test_mixture_distribution():
    mixture = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(logits=[1, 2, 3]),
        components=[
            tfp.distributions.Normal(loc=-1., scale=0.1),
            tfp.distributions.Normal(loc=0., scale=0.30),
            tfp.distributions.Normal(loc=+1., scale=0.5)])

    x = tf.linspace(-2., 3., int(1e4))
    plt.plot(x, mixture.prob(x))
    plt.show()

    approx_entropy = 0.0
    for i, component in enumerate(mixture.components):
        print(f'[{i}] entropy: {component.entropy()}')
        approx_entropy += mixture.cat.prob(i) * component.entropy()
    print('approx entropy:', approx_entropy, mixture.entropy_lower_bound())


def test_mixture_same_family_distribution():
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(logits=[1, 3]),
        components_distribution=tfp.distributions.MultivariateNormalDiag(
            loc=[[0.5, 0.6], [0.1, 0.3]],
            scale_diag=[[0.3, 0.1], [0.6, 0.99]]))

    print(mixture.sample())


def test_compare_continuous_distributions():
    a = [0.1, 0.2, 0.3]
    b = [1.0, 1.0, 1.0]
    a = a[1:]
    b = b[1:]

    distributions = dict(normal=tfp.distributions.Normal(loc=a, scale=b),
                         multivariate_normal=tfp.distributions.MultivariateNormalDiag(loc=a, scale_diag=b),
                         truncated_normal=tfp.distributions.TruncatedNormal(loc=a, scale=b,
                                                                            low=-1.0, high=1.0),
                         beta=tfp.distributions.Beta(b, a),
                         dirichlet=tfp.distributions.Dirichlet(concentration=[1.0, 3]))

    for name, d in distributions.items():
        samples = d.sample(10)
        print(name, ':\n', samples, '\n\tmean:\n', np.mean(samples, axis=1))
        print('prob:')
        print(d.prob(samples))


def test_beta2():
    # a = tf.ones((2, 2))
    # a = tf.ones((2,))
    a = 0.5
    # b = [[1, 2], [2, 1]]
    # b = [[1, 2]]
    b = 0.1
    beta = tfp.distributions.Beta(a, b, validate_args=True)

    samples = beta.sample(5)
    print('samples:', samples)
    print('prob:', beta.prob(samples))
    print('prob:', beta.prob([0.0, 0.01, 0.001, 0.005, 0.1, 1.0, 0.9, 0.95, 0.99, 0.999]))
    print('prob:', beta.log_prob([0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]))
    print('prob:', beta.log_prob([1.0, 0.9, 0.99, 0.999, 0.9999, 0.99999]))


def test_gaussian():
    # normal = tfp.distributions.Normal(loc=[0.0], scale=[1.0])
    normal = tfp.distributions.MultivariateNormalDiag(loc=[0.0, 0.5], scale_diag=[1.0, 1.0])

    samples = normal.sample(20)
    print('samples:', samples)
    print('prob:', normal.prob(samples))

    min_value = -3.0 * normal.stddev()
    max_value = 3.0 * normal.stddev()
    print('support:', min_value.numpy(), max_value.numpy())

    def normalize_01(x):
        r = max_value - min_value
        return (tf.clip_by_value(x, min_value, max_value) - min_value) / r

    print('[0, 1] samples:', normalize_01(samples))


if __name__ == '__main__':
    # Memories:
    # test_recent_memory()
    # test_replay_memory()

    # GAE:

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
    # test_beta(alpha=[[1, 2], [3, 4]], beta=[1, 1])
    # cat = tfp.distributions.Categorical(logits=[1, 2, 3, 4])
    # print(cat.log_prob([[1], [2], [3]]))
    # test_categorical_vs_beta_prob(action_shape=1)
    # test_mixture_distribution()
    # test_mixture_same_family_distribution()
    # test_compare_continuous_distributions()
    test_beta2()
    # test_gaussian()

    # Networks:
    # test_dual_head_value_network()
    # test_dict_inputs()
    # test_add_noise_action()
    # test_noise_lambda_layer()
    # test_predict_different_batch()
    # test_iterate_layer_output()
    # test_model_independent_distribution()
    # test_rnn_sequences()

    # Data:
    # test_tf_data_api()
    # test_space_to_flat_spec()
    # test_gym_spaces_change_shape()
    # test_gym_spaces_expand()
    # test_tf_dataset_shard()
    # test_data_to_batches()
    # test_dataset_sharding_scenarios()

    # test_incremental_mean([1, 2, 3, 3, 0, -1, -6, 13, 2, -7], alpha=0.9)
    # test_incremental_mean([1, 2, 3, 3, 0, -1, -6, 13, 2, -7])
    # test_incremental_mean2([1, 2, 3, 3, 0, -1, -6, 13, 2, -7])
    # test_incremental_std([1, 2, 3, 3, 0, -1, -6, 13, 2, -7])
    pass
