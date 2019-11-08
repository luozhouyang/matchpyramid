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

import argparse
import os

import tensorflow as tf
from easylib.dl import KerasModelDatasetRunner
from nlp_datasets.abstract_dataset import AbstractXYDataset
from nlp_datasets.tokenizers import SpaceTokenizer
from nlp_datasets.xyz_dataset import XYZSameFileDataset

from mp import models, utils

tokenizer = SpaceTokenizer()
tokenizer.build_from_vocab(os.path.join(utils.testdat_dir(), 'vocab.txt'))
config = {
    'x_max_len': 1000,
    'y_max_len': 1000,
    'train_batch_size': 1,
    'predict_batch_size': 32,
    'shuffle_size': -1,
    'num_parallel_calls': tf.data.experimental.AUTOTUNE
}
dataset = XYZSameFileDataset(x_tokenizer=tokenizer, y_tokenizer=tokenizer, config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['indicator', 'dot', 'cosine'], default='indicator')
    parser.add_argument('--action', type=str, default='train', choices=['train', 'eval', 'predict', 'export'])
    parser.add_argument('--model_dir', type=str, default='/tmp/matchpyramid', help='File path to save model.')

    args, _ = parser.parse_known_args()

    if args.model == 'indicator':
        model = models.build_indicator_model(models.model_config)
    elif args.model == 'dot':
        model = models.build_dot_model(models.model_config)
    elif args.model == 'cosine':
        model = models.build_cosine_model(models.model_config)
    else:
        raise ValueError('Invalid model: %s' % args.model)

    runner = KerasModelDatasetRunner(
        model=model,
        model_name='mp',
        model_dir=args.model_dir,
        configs=None
    )

    if args.action == 'train':
        train_files = [os.path.join(utils.testdat_dir(), 'train.txt')]
        # use train files as validation files, not recommend in actual use
        valid_files = [os.path.join(utils.testdat_dir(), 'train.txt')]
        train_dataset = dataset.build_train_dataset(train_files)
        valid_dataset = dataset.build_eval_dataset(valid_files)
        runner.train(dataset=train_dataset, val_dataset=valid_dataset, ckpt=args.model_dir)
    elif args.action == 'eval':
        eval_files = [os.path.join(utils.testdat_dir(), 'train.txt')]
        eval_dataset = dataset.build_eval_dataset(eval_files)
        runner.eval(dataset=eval_dataset)
    elif args.action == 'predict':
        predict_files = [os.path.join(utils.testdat_dir(), 'train.txt')]
        predict_dataset = dataset.build_predict_dataset(predict_files)
        runner.predict(dataset=predict_dataset)
    elif args.action == 'export':
        runner.export(path=os.path.join(args.model_dir, 'export'), ckpt=None)
    else:
        raise ValueError('Invalid action: %s' % args.action)
