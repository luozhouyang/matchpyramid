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

from easylib.dl import KerasModelDatasetRunner

from mp import datas
from mp import models
from mp import utils

MODEL_DIR = '/tmp/matchpyramid'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def train(model):
    r = KerasModelDatasetRunner(
        model,
        model_dir=MODEL_DIR,
        model_name='mp',
        logger_name='matchpyramid')
    train_files = [os.path.join(utils.testdat_dir(), 'train.txt')]
    valid_files = [os.path.join(utils.testdat_dir(), 'vocab.txt')]
    train_dataset = datas.build_train_dataset(train_files, datas.dataset_config)
    valid_dataset = datas.build_eval_dataset(valid_files, datas.dataset_config)
    r.train(dataset=train_dataset, val_dataset=valid_dataset, ckpt=MODEL_DIR)


def eval(model):
    r = KerasModelDatasetRunner(
        model,
        model_dir=MODEL_DIR,
        model_name='mp',
        logger_name='matchpyramid')
    test_files = []
    test_dataset = datas.build_eval_dataset(test_files, datas.dataset_config)
    r.eval(test_dataset, MODEL_DIR)


def predict(model):
    r = KerasModelDatasetRunner(
        model,
        model_dir=MODEL_DIR,
        model_name='mp',
        logger_name='matchpyramid')
    predict_files = []
    pred_dataset = datas.build_predict_dataset(predict_files, datas.dataset_config)
    r.predict(pred_dataset)


def export(model):
    r = KerasModelDatasetRunner(
        model,
        model_dir=MODEL_DIR,
        model_name='mp',
        logger_name='matchpyramid')
    p = ''
    ckpt = None
    r.export(path=p, ckpt=ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['indicator', 'dot', 'cosine'], default='indicator')
    parser.add_argument('action', type=str, default='train', choices=['train', 'eval', 'predict', 'export'])

    args, _ = parser.parse_known_args()

    if args.model == 'indicator':
        model = models.build_indicator_model(models.model_config)
    elif args.model == 'dot':
        model = models.build_dot_model(models.model_config)
    elif args.model == 'cosine':
        model = models.build_cosine_model(models.model_config)
    else:
        raise ValueError('Invalid model: %s' % args.model)

    if args.action == 'train':
        train(model)
    elif args.action == 'eval':
        eval(model)
    elif args.action == 'predict':
        predict(model)
    elif args.action == 'export':
        export(model)
    else:
        raise ValueError('Invalid action: %s' % args.action)
