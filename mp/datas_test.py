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
import os
from mp import datas
from mp import utils

data_file = os.path.join(utils.testdat_dir(), 'train.txt')
vocab_file = os.path.join(utils.testdat_dir(), 'vocab.txt')
print(data_file)

datas.dataset_config['vocab_file'] = vocab_file


class DatasTest(tf.test.TestCase):

    def testBuildTrainDataset(self):
        train_files = [
            data_file
        ]
        dataset = datas.build_train_dataset(train_files, datas.dataset_config)
        v = next(iter(dataset))
        (q, d), l = v
        print('query: \n', q)
        print('doc: \n', d)
        print('label: \n', l)

    def testBuildEvalDataset(self):
        eval_files = [
            data_file
        ]
        dataset = datas.build_eval_dataset(eval_files, datas.dataset_config)
        v = next(iter(dataset))
        (q, d), l = v
        print('query: \n', q)
        print('doc: \n', d)
        print('label: \n', l)

    def testBuildPredictDataset(self):
        predict_files = [
            data_file
        ]
        dataset = datas.build_predict_dataset(predict_files, datas.dataset_config)
        v = next(iter(dataset))
        (q, d) = v
        print('query: \n', q)
        print('doc: \n', d)


if __name__ == '__main__':
    tf.test.main()
