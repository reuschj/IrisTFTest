# Imports
# --------------------------------------

# Misc imports
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.compat.v2.keras.optimizers import SGD

# --------------------------------------

# A sequential nueral network model with dense layers to classify iris sepcies
class IrisModel(tf.keras.Model):

    """A sequential nueral network model with dense layers to classify iris sepcies"""

    def __init__(self, hidden_size=10, feature_size=4, class_count=3):
        super(IrisModel, self).__init__(name='iris_model')
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.class_count = class_count
        # Layers
        self.input_layer = layers.Dense(feature_size, activation='relu')
        self.hidden_layer_01 = layers.Dense(hidden_size, activation='relu')
        self.hidden_layer_02 = layers.Dense(hidden_size, activation='relu')
        self.output_layer = layers.Dense(class_count, activation='softmax')

    def call(self, data):
        x = self.input_layer(data)
        y = self.hidden_layer_01(x)
        z = self.hidden_layer_02(y)
        return self.output_layer(z)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.class_count
        return tf.TensorShape(shape)