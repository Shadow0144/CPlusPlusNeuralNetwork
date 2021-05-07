# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:18:21 2021

@author: Corbi
"""

import tensorflow as tf

input_image = tf.constant([[[[1.], [1.], [2.]],
                           [[2.], [2.], [3.]],
                           [[4.], [1.], [1.]]]])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   input_shape=(3,3,1)))
model.compile('adam', 'mean_squared_error')
output = model.predict(input_image, steps=1)