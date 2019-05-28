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

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from mp.indicator import Indicator


class IndicatorTest(tf.test.TestCase):

    def testIndicator(self):
        indicator = Indicator(100, 100)
        q = np.array([[1, 2, 3], [4, 5, 6]])
        d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        m = indicator((q, d))
        print(m[0], m[1])
        lam = keras.layers.Lambda(lambda x: x + x)(m)
        print(lam)


if __name__ == '__main__':
    tf.test.main()
