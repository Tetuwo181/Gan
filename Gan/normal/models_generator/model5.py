# -*- coding: utf-8 -*-
from keras import Sequential
from keras import Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dropout
from keras import Input
import keras.engine.training
from typing import Union
from typing import Tuple
from util_types import types_of_gan
from keras.layers import LeakyReLU

def builder(
            img_size: types_of_gan.input_img_size = 28,
            channels: int = 3,
            mid_neuro_num: int = 128,
            z_dim: int = 100,
            momentum: float = 0.8,
            kernel_size: Union[int, Tuple[int, int]] = 3
            )->keras.engine.training.Model:
    """
    Ganのgenerator部を作成する。model3の出力をtanhに変更
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

    model.add(Dense(mid_neuro_num*raw*col, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Reshape((raw, col, mid_neuro_num)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(UpSampling2D())
    model.add(Conv2D(mid_neuro_num, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=momentum))
    model.add(UpSampling2D())
    model.add(Dropout(0.5))
    model.add(Conv2D(int(mid_neuro_num/2), kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Conv2D(channels, kernel_size=kernel_size, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)
