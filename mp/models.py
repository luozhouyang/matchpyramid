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

from tensorflow.python import keras

from mp.indicator import Indicator

model_config = {
    'query_max_len': 1000,
    'doc_max_len': 1000,
    'num_conv_layers': 3,
    'filters': [8, 16, 32],
    'kernel_size': [[5, 5], [3, 3], [3, 3]],
    'pool_size': [[2, 2], [2, 2], [2, 2]],
    'dropout': 0.5,
    'batch_size': 32,
    'vocab_size': 100,  # Important!!! update vocab_size
    'embedding_size': 128,
}


def build_dot_model(config):
    """Using dot-product to produce match matrix, as described in the paper."""
    q_input = keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')

    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    # dot
    dot = keras.layers.Dot(axes=-1, name='dot')([q_embedding, d_embedding])
    # reshape to [batch_size, query_max_len, doc_max_len, channel(1)]
    matrix = keras.layers.Reshape((config['query_max_len'], config['doc_max_len'], 1), name='matrix')(dot)

    x = matrix
    for i in range(config['num_conv_layers']):
        x = keras.layers.Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' % i)(x)
        x = keras.layers.MaxPooling2D(pool_size=tuple(config['pool_size'][i]), name='max_pooling_%d' % i)(x)
        x = keras.layers.BatchNormalization()(x)

    flatten = keras.layers.Flatten()(x)
    dense = keras.layers.Dense(32, activation='relu')(flatten)
    out = keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = keras.Model(inputs=[q_input, d_input], outputs=[matrix, out])
    model.compile(
        loss={
            'out': 'binary_crossentropy'
        },
        optimizer='sgd',
        metrics={
            'out': [keras.metrics.Accuracy(), keras.metrics.Recall(), keras.metrics.Precision()]
        })
    return model


def build_cosine_model(config):
    """Using cosine to produce match matrix, as described in the paper."""
    q_input = keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    embedding = keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')

    q_embedding = embedding(q_input)
    d_embedding = embedding(d_input)

    # cosine
    cosine = keras.layers.Dot(axes=-1, normalize=True, name='cosine')([q_embedding, d_embedding])
    matrix = keras.layers.Reshape((config['query_max_len'], config['doc_max_len'], 1), name='matrix')(cosine)

    x = matrix
    for i in range(config['num_conv_layers']):
        x = keras.layers.Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' % i)(x)
        x = keras.layers.MaxPooling2D(pool_size=tuple(config['pool_size'][i]), name='max_pooling_%d' % i)(x)
        x = keras.layers.BatchNormalization()(x)

    flatten = keras.layers.Flatten()(x)
    dense = keras.layers.Dense(32, activation='relu')(flatten)
    out = keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = keras.Model(inputs=[q_input, d_input], outputs=[matrix, out])
    model.compile(
        loss={
            'out': 'binary_crossentropy'
        },
        optimizer='sgd',
        metrics={
            'out': [keras.metrics.Accuracy(), keras.metrics.Recall(), keras.metrics.Precision()]
        })
    return model


def build_indicator_model(config):
    """Using indicator fn to produce match matrix, as described in the paper."""
    q_input = keras.layers.Input(shape=(config['query_max_len'],), name='q_input')
    d_input = keras.layers.Input(shape=(config['doc_max_len'],), name='d_input')

    m = Indicator(config['query_max_len'], config['doc_max_len'], name='matrix')((q_input, d_input))
    m2 = keras.layers.Reshape((config['query_max_len'], config['doc_max_len'], 1), name='m2')(m)
    x = m2
    for i in range(config['num_conv_layers']):
        x = keras.layers.Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            padding='same',
            activation='relu',
            name='conv_%d' % i)(x)
        x = keras.layers.MaxPooling2D(pool_size=tuple(config['pool_size'][i]), name='max_pooling_%d' % i)(x)
        x = keras.layers.BatchNormalization()(x)

    flatten = keras.layers.Flatten()(x)
    dense = keras.layers.Dense(32, activation='relu')(flatten)
    out = keras.layers.Dense(1, activation='sigmoid', name='out')(dense)

    model = keras.Model(inputs=[q_input, d_input], outputs=[out, m])
    model.compile(
        loss={
            'out': 'binary_crossentropy'
        },
        optimizer='sgd',
        metrics={
            'out': [keras.metrics.Accuracy(), keras.metrics.Recall(), keras.metrics.Precision()]
        })
    return model
