# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras import Input
from typing import Union
from typing import Tuple
from util_types import types_of_gan


def builder(
            img_size: types_of_gan.input_img_size = 28,
            channels: int =1,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            )->Model:
    """
    Ganのdiscriminator部を作成する。model1のパラメータ違い
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :return: discriminator部のモデル
    """

    raw, col = types_of_gan.get_size_pair(img_size)

    img_shape = (raw, col, channels)

    model = Sequential()

    model.add(Conv2D(raw, kernel_size=kernel_size, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Conv2D(raw*2, kernel_size=kernel_size, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(raw*4, kernel_size=kernel_size, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(raw*8, kernel_size=kernel_size, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
