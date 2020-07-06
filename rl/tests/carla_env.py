
from tensorflow.keras.layers import *

from rl.environments import OneCameraCARLAEnvironment, ThreeCameraCARLAEnvironment, CARLAPlayWrapper, \
                            CARLACollectWrapper
from rl.agents import PPOAgent, PPO2Agent
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


# TODO: make "swish" activation (layer)


class CarlaPPOAgent(PPOAgent):

    def __init__(self, *args, name='carla-agent', **kwargs):
        super().__init__(*args, **kwargs)

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


if __name__ == '__main__':
    # 1 camera
    # CARLAPlayWrapper(OneCameraCARLAEnvironment(debug=True, window_size=(600, 450))).play()

    # 3 cameras
    # CARLAPlayWrapper(OneCameraCARLAEnvironment(debug=False, vehicle_filter='vehicle.tesla.model3',
    #                                              window_size=(600, 200))).play()

    # Collect Wrapper
    # CARLACollectWrapper(OneCameraCARLAEnvironment(debug=True, window_size=(600, 450), render=True,
    #                                               image_shape=(150, 200, 1)),
    #                     ignore_traffic_light=True, name='test') \
    #     .collect(episodes=64, timesteps=256, episode_reward_threshold=15.0 * 200)

    # Imitation Learning
    env = OneCameraCARLAEnvironment(debug=True, window_size=(600, 450), render=False,
                                    image_shape=(150, 200, 1))
    agent = CarlaPPOAgent(env)
    agent.summary()

    ImitationWrapper(agent, policy_lr=1e-3, value_lr=1e-4, name='test')\
        .imitate(batch_size=64, shuffle_batches=True, repetitions=2, save_every=16)

    pass
