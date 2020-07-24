
import tensorflow as tf
from tensorflow.keras.layers import *

from rl.environments import OneCameraCARLAEnvironment, ThreeCameraCARLAEnvironment, CARLAPlayWrapper, \
                            CARLACollectWrapper, CARLABenchmark, ThreeCameraCARLAEnvironmentDiscrete
from rl.environments.carla import env_utils as carla_utils

from rl import utils
from rl import augmentations as aug
from rl.agents import PPOAgent
from rl.agents.imitation import ImitationWrapper


def my_cnn(input_layer: Layer, filters=32, dropout=0.2, noise=0.05, layers=(2, 5), normalization='layer',
           activation1='tanh', activation2='elu', filters_multiplier=1, dense_layers=2, dense_activation='relu',
           dense_units=32, dense_dropout=0.5) -> Layer:
    # Normalization
    if normalization == 'layer':
        x = LayerNormalization()(input_layer)
    else:
        x = input_layer

    # Gaussian Noise
    if noise > 0.0:
        x = GaussianNoise(stddev=noise)(x)

    # Part 1: depthwise-conv + conv + max-pool
    for i in range(1, layers[0] + 1):
        x = DepthwiseConv2D(kernel_size=3, padding='same', activation=activation1)(x)
        x = Conv2D(filters=int(filters * i * filters_multiplier), kernel_size=3, padding='same',
                   activation=activation2)(x)
        x = SpatialDropout2D(rate=dropout)(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)  # overlapping max-pool

    # Part 2:
    for i in range(1, layers[1] + 1):
        padding = 'same' if i % 2 == 0 else 'valid'

        x = DepthwiseConv2D(kernel_size=3, padding=padding, activation=activation1)(x)
        x = Conv2D(filters=int(filters * (i + layers[0]) * filters_multiplier), kernel_size=3, padding='valid',
                   activation=activation2)(x)
        x = SpatialDropout2D(rate=dropout)(x)

    x = GlobalAveragePooling2D()(x)

    # Final part: dense layers
    for _ in range(dense_layers):
        x = Dense(units=dense_units, activation=dense_activation)(x)
        x = Dropout(rate=dense_dropout)(x)

    return x


def dense_network(input_layer: Layer, units=32, layers=2, activation='relu', dropout=0.5):
    x = input_layer

    for _ in range(layers):
        x = Dense(units=units, activation=activation)(x)
        x = Dropout(rate=dropout)(x)

    return x


class CarlaPPOAgent(PPOAgent):

    def __init__(self, *args, name='carla-agent', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def policy_layers(self, inputs: dict, **kwargs) -> Layer:
        print(inputs.keys())
        # Retrieve input layers
        image_in = inputs['state_image']
        road_in = inputs['state_road']
        vehicle_in = inputs['state_vehicle']
        command_in = inputs['state_command']
        control_in = inputs['state_past_control']

        # Output layers
        image_out = my_cnn(image_in, filters=8, layers=(5, 0))
        road_out = dense_network(road_in)
        vehicle_out = dense_network(vehicle_in)
        control_out = dense_network(control_in)
        command_out = dense_network(command_in)

        return concatenate([image_out, road_out, vehicle_out, control_out, command_out])

    # def value_layers(self, input_layers: dict, **kwargs):
    #     pass


class CARLAImitationLearning(ImitationWrapper):

    def __init__(self, *args, target_size=None, grayscale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.to_grayscale = grayscale

    def preprocess(self):
        target_size = tf.TensorShape(dims=self.target_size)
        should_grayscale = tf.constant(self.to_grayscale, dtype=tf.bool)

        @tf.function
        def augment_fn(state, _):
            state = state.copy()
            image = state['state_image']
            image = aug.tf_resize(image, size=target_size)

            # contrast, tone, saturation, brightness
            if aug.tf_chance() > 0.5:
                image = aug.tf_saturation(image)

            if aug.tf_chance() > 0.5:
                image = aug.tf_contrast(image, lower=0.5, upper=1.5)

            if aug.tf_chance() > 0.5:
                image = aug.tf_hue(image)

            if aug.tf_chance() > 0.5:
                image = aug.tf_brightness(image, delta=0.5)

            # blur
            if aug.tf_chance() < 0.33:
                image = aug.tf_gaussian_blur(image, size=5)

            # noise
            if aug.tf_chance() < 0.2:
                image = aug.tf_salt_and_pepper(image, amount=0.1)

            if aug.tf_chance() < 0.33:
                image = aug.tf_gaussian_noise(image, amount=0.15, std=0.15)

            image = aug.tf_normalize(image)

            # cutout & dropout
            if aug.tf_chance() < 0.15:
                image = aug.tf_cutout(image, size=6)

            if aug.tf_chance() < 0.10:
                image = aug.tf_coarse_dropout(image, size=49, amount=0.1)

            if should_grayscale:
                image = aug.tf_grayscale(image)

            state['state_image'] = 2.0 * image - 1.0  # -1, +1
            return state, _

        # print(tf.autograph.to_code(augment_fn.python_function))
        return augment_fn


if __name__ == '__main__':
    # 1 camera
    # CARLAPlayWrapper(OneCameraCARLAEnvironment(debug=True, camera='rgb', window_size=(600, 450))).play()

    # 3 cameras
    # CARLAPlayWrapper(ThreeCameraCARLAEnvironment(debug=True, window_size=(720, 320))).play()

    # Collect Wrapper
    # CARLACollectWrapper(ThreeCameraCARLAEnvironment(debug=False, window_size=(600, 450), render=False,
    #                                                 image_shape=(120, 160, 3)),
    #                     ignore_traffic_light=True, name='collect-3camera') \
    #     .collect(episodes=64, timesteps=1000, episode_reward_threshold=15.0 * 900)  # 100 traces

    # Imitation Learning
    # env = OneCameraCARLAEnvironment(debug=True, window_size=(600, 450), render=False,
    #                                 image_shape=(150-15, 200-20, 1))
    # agent = CarlaPPOAgent(env, batch_size=32)
    # agent.summary()
    #
    # CARLAImitationLearning(agent, target_size=(135, 180), policy_lr=1e-3, value_lr=1e-4, name='test-preprocess')\
    #     .imitate(shuffle_batches=True, repetitions=2, save_every=16)

    # Test Benchmark:
    # bench_env = CARLABenchmark(OneCameraCARLAEnvironment(debug=True, camera='rgb'),
    #                            task=CARLABenchmark.Tasks.REGULAR_TRAFFIC,
    #                            weather=CARLABenchmark.TRAIN_WEATHERS)
    #
    # CARLAPlayWrapper(bench_env).play()
    # print(bench_env.success_rate())

    # Test discrete action-space:
    # CARLAPlayWrapper(ThreeCameraCARLAEnvironmentDiscrete(bins=8, debug=True, window_size=(720, 320)))\
    #     .play()
    pass
