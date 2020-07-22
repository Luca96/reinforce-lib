"""1-Step (online) Recurrent Neural Network Layer"""

import tensorflow as tf

from tensorflow.keras.layers import *


# TODO: serialization!
# TODO: define convolutional and stacked versions as well

# class OnlineLSTM(LSTMCell):
#     """Stateful 1-step, online LSTM Layer"""
#
#     def __init__(self, units: int, *args, **kwargs):
#         super().__init__(units, *args, **kwargs)
#
#         # states
#         self.hidden_state = tf.Variable(initial_value=tf.zeros((1, units)), trainable=False)
#         self.cell_state = tf.Variable(initial_value=tf.zeros((1, units)), trainable=False)
#
#     def __call__(self, inputs, training=None, **kwargs):
#         h, [h, c] = super().__call__(inputs, states=[self.hidden_state, self.cell_state], training=training)
#
#         # update state
#         self.hidden_state.assign(tf.expand_dims(h[-1], axis=0))
#         self.cell_state.assign(tf.expand_dims(c[-1], axis=0))
#
#         return h
#
#     def get_state(self):
#         return tf.identity(self.hidden_state), tf.identity(self.cell_state)
#
#     def reset_state(self):
#         self.hidden_state.assign(tf.zeros_like(self.hidden_state))
#         self.cell_state.assign(tf.zeros_like(self.cell_state))


class OnlineGRU(GRUCell):
    """Stateful 1-step, online GRU Layer"""

    def __init__(self, units: int, *args, depth=0, depth_activation='swish', **kwargs):
        super().__init__(units, *args, **kwargs)

        self.state = tf.Variable(initial_value=tf.zeros((1, units)), trainable=False)
        self.depth = depth

        if depth > 0:
            self.layers = [Dense(units, activation=depth_activation) for _ in range(depth)]

    def call(self, inputs, **kwargs):
        h, [h] = super().call(inputs, states=[self.state], **kwargs)

        # update state
        if self.depth > 0:
            for layer in self.layers:
                h = layer(h)

        self.state.assign(tf.expand_dims(h[-1], axis=0))
        return h

    def get_state(self):
        return tf.identity(self.state)

    def reset_state(self):
        self.state.assign(tf.zeros_like(self.state))


class StackedOnlineGRU(StackedRNNCells):
    def __init__(self, units: int, num_cells: int, **kwargs):
        assert num_cells > 0

        self.cells = [OnlineGRU(units, **kwargs) for _ in range(num_cells)]
        self.state = [cell.state for cell in self.cells]

        super().__init__(cells=self.cells)

    # def call(self, inputs, states, constants=None, training=None, **kwargs):
    #     tf.print(kwargs)
    #     tf.print(states)
    #     h, states = super().call()
    #
    #     return h

    def get_state(self):
        return [cell.get_state() for cell in self.cells]

    def reset_state(self):
        for cell in self.cells:
            cell.reset_state()
