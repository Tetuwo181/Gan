# -*- coding: utf-8 -*-
from keras import Model
from keras import Input
from keras.models import Sequential
import keras.engine.training

from typing import Callable
MODEL = keras.engine.training.Model  # 型アノテーションのために別名つけただけ


def combine_type1(generator: Model,  discriminator: Model)->Model:
    discriminator.trainable = False
    return Sequential(generator, discriminator)


def combine_type2(generator: Model, discriminator: Model, z_dim: int = 100)->Model:
    """
    モデルをマージする
    :param generator:　generatorネットワーク
    :param discriminator: discriminatorネットワーク
    :param z_dim: 潜在次元
    :return: 入力として潜在変数(Noise)noise_inputと出力にReal(=1), Fake(=0)を取る
    """
    z = Input(shape=(z_dim,))
    img = generator(z)
    # discriminatorのパラメータを固定
    discriminator.trainable = False
    valid = discriminator(img)
    model = Model(z, valid)
    model.summary()
    return model


def combine_builder(init_type: int = 0, z_dim: int = 100)->Callable[[Model, Model], Model]:
    def combine(generator, discriminator): return combine_type2(generator, discriminator, z_dim)
    return combine if init_type == 0 else combine_type1

