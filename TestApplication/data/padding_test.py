# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:51:27 2021

@author: Corbi
"""

import tensorflow as tf

input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', padding="same", input_shape=input_shape[1:], strides=2)(x)
print(y.shape)