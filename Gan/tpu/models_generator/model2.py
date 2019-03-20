# -*- coding: utf-8 -*-
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import Input
from typing import Union
from typing import Tuple
from util_types import types_of_gan


def builder(
            img_size: types_of_gan.input_img_size = 28,
            channels: int = 3,
            mid_neuro_num: int = 128,
            z_dim: int = 100,
            momentum: float = 0.8,
            kernel_size: Union[int, Tuple[int, int]] = 3
            )->Model:
    """
    Ganのgenerator部を作成する
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, col)
    :param channels:色の出力変数（白黒画像なら1）
    :param mid_neuro_num: 中間層のニューロの数のベース
    :param z_dim:潜在変数:(generatorの入力)
    :param momentum:Batchの移動平均のためのMomentum
    :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :return: generator部のモデル
    """

    raw, col = types_of_gan.get_size_pair(img_size)
    noise_shape = (z_dim,)

    model = Sequential()

    model.add(Dense(mid_neuro_num*raw*col, activation="relu", input_shape=noise_shape))
    model.add(Reshape((raw, col, mid_neuro_num)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(UpSampling2D())
    model.add(Conv2D(mid_neuro_num, kernel_size=kernel_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(UpSampling2D())
    model.add(Conv2D(int(mid_neuro_num/2), kernel_size=kernel_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Conv2D(channels, kernel_size=kernel_size, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)
