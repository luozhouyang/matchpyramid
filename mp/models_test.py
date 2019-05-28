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

from mp import models


class ModelsTest(tf.test.TestCase):

    def testBuildIndicatorModel(self):
        m = models.build_indicator_model(models.model_config)
        m.summary()

    def testBuildDotModel(self):
        m = models.build_dot_model(models.model_config)
        m.summary()

    def testBuildCosineModel(self):
        m = models.build_cosine_model(models.model_config)
        m.summary()


if __name__ == '__main__':
    tf.test.main()
