# -*- coding: utf-8 -*-
from Gan.normal import discriminator as ds, generator as gn
from keras import Model
from keras import Input
from keras.models import Sequential
import keras.engine.training
from keras import optimizers
from DataIO import data_loader
from DataIO import data_saver
from DataIO import create_dir
import numpy as np
import os
import cv2
from typing import Optional
from typing import Callable
from typing import Tuple
from typing import List
from numba import jit
from util_types import types_of_gan
from Gan.normal.gan import Gan


class GanWithClasses(Gan):
    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 class_set: List[int],
                 optimizer: types_of_gan.GanOptimizer = optimizers.Adam(0.0002, 0.5),
                 combine_type: int = 0
                 ):
        super().__init__(generator, discriminator, optimizer, combine_type)
        self.__class_set = class_set
        self.__class_num = len(self.__class_set)

    @jit
    def generate_fake(self, build_num: int, class_index: int)-> np.ndarray:
        """
        generatorから自動生成されたデータを作成する
        :param build_num: 自動生成するデータ数
        :return: 自動生成された画像データ
        """
        class_info = np.array([1 if index == class_index else 0 for index in self.__class_set])
        noise = np.random.normal(0, 1, (build_num, self.z_dim- self.__class_num))
        input_value = np.appemd(class_info, noise)
        return self.__generator.predict(input_value)

    @jit
    def train_discriminator(self,
                            half_batch: int,
                            img_set: np.ndarray,
                            class_value:int,
                            real_label_noise: Optional[float] = None,
                            fake_label_noise: Optional[float] = None,
                            prob_replace: float = 0)->Tuple[float, float]:
        """
        discriminator部の学習
        :param half_batch: バッチサイズの半分 半分を実際のデータセットから復元抽出したもの　半分を生成されたデータを用いて学習する
        :param img_set: 実際のデータセット
        :param real_label_noise: 本物データに対するラベルのノイズの範囲
        :param fake_label_noise: 偽物データに対するラベルのノイズの範囲
        :param prob_replace: 本物のラベルと偽物のラベルの置き換え確立
        :return: 本物のデータと偽物のデータの損失率
        """
        real_image_set = data_loader.sampling_real_data_set(half_batch, img_set)
        fake_image_set = self.generate_fake(half_batch)
        real_label_base = self.build_real_label(half_batch, real_label_noise)
        fake_label_base = self.build_fake_label(half_batch, fake_label_noise)
        base_mask, replaced_mask = self.build_label_replacer(half_batch, prob_replace)
        real_label = real_label_base*base_mask + fake_label_base*replaced_mask
        fake_label = fake_label_base*base_mask + real_label_base*replaced_mask
        loss_real = self.__discriminator.train_on_batch(real_image_set, real_label)
        loss_fake = self.__discriminator.train_on_batch(fake_image_set, fake_label)
        return loss_real, loss_fake

    @jit
    def train_generator(self, batch_size: int):
        """
        generatorの学習
        :param batch_size: バッチサイズ
        :return: 学習した際の損失率
        """
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        # 生成データの正解ラベルは本物（1）
        valid_y = np.array([1] * batch_size)
        return self.__combined.train_on_batch(noise, valid_y)
