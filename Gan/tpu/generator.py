# -*- coding: utf-8 -*-
from tensorflow.keras import Model
from typing import Union
from typing import Tuple
from util_types import types_of_gan
import importlib


def builder(
            img_size: types_of_gan.input_img_size = 28,
            channels: int = 3,
            mid_neuro_num: int = 128,
            z_dim: int = 100,
            momentum: float = 0.8,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            model_name: str = "model1"
            )->Model:
    """
    Ganのgenerator部を作成する
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, col)
    :param channels:色の出力変数（白黒画像なら1）
    :param mid_neuro_num: 中間層のニューロの数のベース
    :param z_dim:潜在変数:(generatorの入力)
    :param momentum:Batchの移動平均のためのMomentum
    :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :param model_name: インポートするモデルの名前。models_generator以下のディレクトリにモデル生成器を置く
    :return: generator部のモデル
    """

    model_module = importlib.import_module("Gan.tpu.models_generator."+model_name)
    return model_module.builder(img_size, channels, mid_neuro_num, z_dim, momentum, kernel_size)


def get_z_dim(generator_model:Model)-> int:
    """
    特徴変数の値をモデルから取得する
    :param generator_model: 取得対象となるモデル
    :return: 特徴変数
    """
    return generator_model.input.shape.as_list()[-1]


def init_input_image(size: types_of_gan.input_img_size):
    def builder_of_generator(channels: int = 1,
                             mid_neuro_num: int = 128,
                             z_dim: int = 100,
                             momentum: float = 0.8,
                             kernel_size: Union[int, Tuple[int, int]] = 3):
        """
        Ganのgenerator部を作成する
        :param channels:色の出力変数（白黒画像なら1）
        :param mid_neuro_num: 中間層のニューロの数のベース
        :param z_dim:潜在変数:(generatorの入力)
        :param momentum:Batchの移動平均のためのMomentum
        :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
        :return: generator部のモデル
        """
        return builder(size, channels, mid_neuro_num, z_dim, momentum, kernel_size)
    return builder_of_generator
