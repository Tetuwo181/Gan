# -*- coding: utf-8 -*-
"""
kerasの最適化関数のクラスの型を記述
"""
import tensorflow as tf
from tensorflow.keras import optimizers
from typing import Union

type_optimizer = Union[optimizers.Adam,
                       optimizers.SGD,
                       optimizers.RMSprop,
                       optimizers.Adagrad,
                       optimizers.Nadam,
                       optimizers.TFOptimizer]

