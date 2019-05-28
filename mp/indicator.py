# Copyright 2019 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.python import keras


class Indicator(keras.layers.Layer):
    """Indicator function to produce match matrix. As described in paper https://arxiv.org/abs/1606.04648"""

    def __init__(self, width, height, **kwargs):
        """Init.

        Args:
            width: width of the matrix
            height: height of the matrix
        """
        super(Indicator, self).__init__(kwargs)
        self.width = width
        self.height = height

    def call(self, inputs, **kwargs):
        """forward pass.

        Args:
            inputs: query and doc. The shape of query and doc: [batch_size, seq_length]

        Returns:
            A tensor. Shape: [batch_size, width, height]
        """
        q, d = inputs
        q_paddings = tf.constant([[0, 0], [0, self.width]])
        q_pad = tf.pad(q, q_paddings, mode='constant', constant_values=-1)
        q_pad = tf.slice(q_pad, [0, 0], [-1, self.width, ])
        q_pad = tf.expand_dims(q_pad, -1)
        q_pad = tf.tile(q_pad, [1, 1, self.height])

        d_paddings = tf.constant([[0, 0], [0, self.height]])
        d_pad = tf.pad(d, d_paddings, mode='constant', constant_values=-2)
        d_pad = tf.slice(d_pad, [0, 0], [-1, self.height])
        d_pad = tf.expand_dims(d_pad, -1)
        d_pad = tf.tile(d_pad, [1, 1, self.width])
        d_pad = tf.transpose(d_pad, perm=[0, 2, 1])

        m = tf.cast(tf.equal(q_pad, d_pad), dtype=tf.float32)
        return m

    def compute_output_shape(self, input_shape):
        """output shape: [batch_size, self.width, self.height]"""
        q_shape, d_shape = input_shape
        assert q_shape[0] <= self.width
        assert d_shape[0] <= self.height
        return (self.width, self.height)
