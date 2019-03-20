# -*- coding: utf-8 -*-
import tensorflow as tf
import keras.engine.training
from typing import Union
from typing import Tuple
from util_types import types_of_gan
import importlib


def builder(
            img_size: types_of_gan.input_img_size = 28,
            channels: int = 3,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            model_name: str = "model1"
            )->keras.engine.training.Model:
    """
    Ganのdiscriminator部を作成する
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :param model_name: インポートするモデルの名前。models_discriminator以下のディレクトリにモデル生成器を置く
    :return: discriminator部のモデル
    """
    model_module = importlib.import_module("Gan.normal.models_discriminator."+model_name)
    return model_module.builder(img_size, channels, kernel_size)


def init_input_image(size: types_of_gan.input_img_size):
    def builder_of_generator(channels: int =1, kernel_size: Union[int, Tuple[int, int]] = 3):
        """
        Ganのgenerator部を作成する
        :param channels:色の出力変数（白黒画像なら1）
        :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
        :return: discriminator部のモデル
        """
        return builder(size, channels, kernel_size)
    return builder_of_generator
