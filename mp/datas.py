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

import os
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from mp import utils

dataset_config = {
    'shuffle_size': 5000000,
    'num_parallel_calls': 8,
    'query_max_len': 1000,
    'doc_max_len': 1000,
    'batch_size': 32,
    'predict_batch_size': 32,
    'vocab_file': os.path.join(utils.testdat_dir(), 'vocab.txt'), # used to build a lookup table to convert chars(words) to ids.
}

# first word(index=0) in vocab_file is `<unk>`
str2id = lookup_ops.index_table_from_file(dataset_config['vocab_file'], default_value=0)
# not used
# id2str = lookup_ops.index_to_string_table_from_file(dataset_config['vocab_file'], default_value='<unk>')

unk_id = tf.constant(0, dtype=tf.int64)

# only used in indicator model's dataset, otherwise, suggested to set these two values to unk_id
query_padding_value = tf.constant(-1, dtype=tf.int64)
doc_padding_value = tf.constant(-2, dtype=tf.int64)


@tf.function
def normalize_label_fn(x):
    if x == '0':
        return 0
    else:
        return 1


def _common_process_dataset(dataset, config):
    dataset = dataset.shuffle(config['shuffle_size'])
    dataset = dataset.filter(lambda x: tf.equal(tf.size(tf.string_split([x], delimiter='@@@').values), 3))
    dataset = dataset.map(
        lambda x: (tf.string_split([x], delimiter='@@@').values[0],
                   tf.string_split([x], delimiter='@@@').values[1],
                   tf.string_split([x], delimiter='@@@').values[2]),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda q, d, l: (tf.string_split([q], delimiter=' ').values,
                         tf.string_split([d], delimiter=' ').values,
                         tf.strings.to_number(l, out_type=tf.int32)),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda q, d, l: (q[:config['query_max_len']], d[:config['doc_max_len']], l),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda q, d, l: (str2id.lookup(q), str2id.lookup(d), l),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def _build_dataset(dataset, config):
    dataset = _common_process_dataset(dataset, config)
    dataset = dataset.padded_batch(
        batch_size=config['batch_size'],
        padded_shapes=(config['query_max_len'],
                       config['doc_max_len'],
                       []),
        padding_values=(query_padding_value, doc_padding_value, 0)  # you can change the padding values
    )
    dataset = dataset.map(
        lambda q, d, l: ((q, d), l),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def build_train_dataset(train_files, config):
    dataset = tf.data.Dataset.from_tensor_slices(train_files)
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(config.get('skip_count', 0)))
    dataset = _build_dataset(dataset, config)
    return dataset


def build_eval_dataset(eval_files, config):
    dataset = tf.data.Dataset.from_tensor_slices(eval_files)
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x))
    dataset = _build_dataset(dataset, config)
    return dataset


def build_predict_dataset(predict_files, config):
    """假设predict文件也带有label"""
    dataset = tf.data.Dataset.from_tensor_slices(predict_files)
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(config.get('skip_count', 0)))
    dataset = _common_process_dataset(dataset, config)
    dataset = dataset.map(
        lambda q, d, l: (q, d),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(
        batch_size=config['predict_batch_size'],
        padded_shapes=(config['query_max_len'],
                       config['doc_max_len']),
        padding_values=(unk_id, unk_id)
    )
    dataset = dataset.map(lambda q, d: ((q, d)))
    return dataset
