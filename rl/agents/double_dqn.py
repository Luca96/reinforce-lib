"""Double Deep Q-Learning (DDQN) Agent"""

import tensorflow as tf


from rl.agents import DQNAgent


class DoubleDQN(DQNAgent):

    def __init__(self, *args, name='double-dqn-agent', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def targets(self, next_states, rewards, terminals):
        """Compute targets using the `Double Q-Learning formula` to reduce overestimation of Q-values"""
        # a = argmax(St+1, a')
        next_q = self.dqn.q_values(next_states, training=True)
        action = tf.cast(tf.argmax(next_q, axis=1), dtype=tf.float32)
        action = tf.expand_dims(action, axis=-1)

        # Q(S_t+1, a)
        q_values = self.target.q_values(next_states, training=False)
        q_values = self.index_q_values(q_values, action)

        # y = r + gamma * q
        targets = rewards + self.gamma * q_values
        targets = tf.where(terminals == 0.0, x=rewards, y=targets)
        return tf.stop_gradient(targets)
