# -*- coding: utf-8 -*-
from Gan.normal import discriminator as ds, generator as gn
from Gan.normal import gan_convinor
from keras import Model
import keras.engine.training
from keras import optimizers
from DataIO import data_loader
from DataIO import data_saver
from DataIO import create_dir
import numpy as np
import os
import cv2
from typing import Optional
from typing import Tuple
from numba import jit
from util_types import types_of_gan

RESULT_ROOT_BASE = os.path.join(os.getcwd(), "result")
MODEL = keras.engine.training.Model  # 型アノテーションのために別名つけただけ


def init_input_dim(size: types_of_gan.input_img_size):
    return gn.init_input_image(size), ds.init_input_image(size)


class Gan(object):
    """
    Ganネットワークを管理するクラス
    他のサンプルコードだとこのクラスで設定を行っているけど
    しっくりこなかったからgeneratorとdiscriminatorのネットワークは別なところで生成し、コンストラクタに渡す
    """
    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 optimizer: types_of_gan.GanOptimizer = optimizers.Adam(0.0002, 0.5),
                 combine_type: int = 0
                 ):
        """
        コンストラクタ内でgeneratorとdiscriminatorをマージする
        :param generator: generatorネットワーク
        :param discriminator: discriminatorネットワーク
        :param optimizer: 最適化関数
        :param combine_type: モデルをマージする手法
        :param tpu_grpc_url: TPUのURL
        """
        discriminator_optimizer, generator_optimizer = types_of_gan.get_optimizer_set(optimizer)
        self.__image_size = tuple(discriminator.input.shape.as_list()[1:3])

        # discriminatorの設定
        self.__discriminator = discriminator
        self.__discriminator.compile(loss='binary_crossentropy',
                                     optimizer=discriminator_optimizer,
                                     metrics=['accuracy'])

        # Generator学習用のCombined_modelを作成
        self.__generator = generator
        self.__z_dim = gn.get_z_dim(generator)
        self.__combined = gan_convinor.combine_builder(combine_type, self.__z_dim)(generator, discriminator)
        self.__combined.compile(loss='binary_crossentropy', optimizer=generator_optimizer)


    @property
    def z_dim(self)->int:
        return self.__z_dim

    @property
    def image_size(self):
        """
        入力サイズ(pixel)
        読み込んだデータはこのサイズにリサイズされて入力される
        :return:
        """
        return self.__image_size

    @property
    def combined_model(self):
        return self.__combined

    @jit
    def generate_fake(self, build_num: int)-> np.ndarray:
        """
        generatorから自動生成されたデータを作成する
        :param build_num: 自動生成するデータ数
        :return: 自動生成された画像データ
        """
        noise = np.random.normal(0, 1, (build_num, self.z_dim))
        return self.__generator.predict(noise)

    @jit
    def train_discriminator(self,
                            half_batch: int,
                            img_set: np.ndarray,
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

    def build_real_label(self, half_batch:int, real_noise:Optional[float] = None):
        if real_noise is None:
            return np.ones((half_batch, 1))
        label_base = np.ones((half_batch, 1))
        return label_base + real_noise * (np.random.rand(half_batch, 1)-0.5)

    def build_fake_label(self, half_batch: int, fake_noise: Optional[float] = None):
        if fake_noise is None:
            return np.zeros((half_batch, 1))
        label_base = np.zeros((half_batch, 1))
        return label_base + fake_noise * np.random.rand(half_batch, 1)

    def build_label_replacer(self, half_batch:int, prob_replace: float = 0):
        base_mask = np.array([[0 if np.random.rand() < prob_replace else 1] for i in range(half_batch)], dtype=float)
        replaced_mask = np.abs(1-base_mask)
        return base_mask, replaced_mask

    @jit
    def train(self,
              data_set: np.ndarray,
              iteration_num: int = 100,
              batch_size: int = 128,
              real_label_noise: Optional[float] = None,
              fake_label_noise: Optional[float] = None,
              prob_replace: float = 0,
              rate_g_and_d: int = 1,
              save_interval: int = 50,
              show_raw: int = 5,
              show_col: int = 5,
              noise_base: Optional[np.ndarray] = None,
              result_root: str = RESULT_ROOT_BASE
              ):
        """
        ネットワーク全体の学習を行う
        :param data_set: 学習用に使用される実データ
        :param iteration_num: 学習を繰り返す回数
        :param batch_size: バッチサイズ　この半数を実際のデータセットから復元抽出、残りの半分を自動生成
        :param real_label_noise: 本物データに対するラベルのノイズの範囲
        :param fake_label_noise: 偽物データに対するラベルのノイズの範囲
        :param prob_replace: 本物と偽物のラベルを入れ替える確率
        :param save_interval: 画像をセーブするくり返し回数の間隔
        :param rate_g_and_d discriminatorを1回学習するごとにgeneratorを学習させる回数
        :param show_raw: 画像を表示する際の横の数
        :param show_col: 画像を表示する際の縦の数
        :param noise_base: 画像を記録する際に生成するノイズのベース　Noneを引数として渡した場合、毎回ランダムに
        :param result_root: 生成されたデータを保存する際のルートディレクトリ
        :return:
        """

        create_dir.create(result_root)

        half_batch = int(batch_size/2)
        network_result_dir_path = os.path.join(result_root, "network_iter"+str(iteration_num))
        create_dir.create(network_result_dir_path)
        result_dir_path = os.path.join(result_root, "result")
        create_dir.create(result_dir_path)
        print("start training")
        for iteration in range(iteration_num):
            # Discriminatorの学習
            d_loss_real, d_loss_fake = self.train_discriminator(half_batch,
                                                                data_set,
                                                                real_label_noise,
                                                                fake_label_noise,
                                                                prob_replace)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Generatorの学習
            for i in range(rate_g_and_d):
                g_loss = self.train_generator(batch_size)
                print("iteration %d_%d [G loss: %f]" % (iteration, i, g_loss))
            print("iteration %d [D loss: %f, acc.: %.2f%%] " % (iteration, d_loss[0], 100 * d_loss[1]))
            if iteration % save_interval == 0:
                noise = np.random.uniform(-1, 1, (show_raw * show_col, self.z_dim)) if noise_base is None else noise_base
                self.save_images_by_grid(iteration, noise, show_raw, show_col, result_dir_path)
                start = np.expand_dims(noise[0], axis=0)
                end = np.expand_dims(noise[1], axis=0)
                result_image = self.visualize_interpolation(start=start, end=end)
                cv2.imwrite(result_dir_path+"latent_{}.png".format(iteration), result_image)
                self.save_network(network_result_dir_path, iteration)

    @jit
    def save_images_by_grid(self,
                            iteration: int,
                            noise: np.ndarray,
                            show_raw: int = 5,
                            show_cal: int = 5,
                            result_root: str = RESULT_ROOT_BASE
                            ):
        gen_images = 0.5*self.__generator.predict(noise)+0.5
        result_path = os.path.join(result_root, str(iteration) + ".png")
        data_saver.save_by_grid(gen_images, result_path, show_raw, show_cal)

    def visualize_interpolation(self, start, end, nb_steps=10):
        alpha_values = np.linspace(0, 1, nb_steps)
        vectors = np.array([start * (1 - alpha) + end * alpha for alpha in alpha_values])

        def img_converter(gen_img_base): return (0.5*gen_img_base+0.5)*255
        gen_img_set = [img_converter(np.squeeze(self.__generator.predict(vec), axis=0)) for vec in vectors]

        result_image = None

        for gen_img in gen_img_set:
            interpolated_image = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR).astype(np.uint8)
            result_image = interpolated_image if result_image is None else np.hstack([result_image, interpolated_image])
        return result_image

    @jit
    def save_network(self,
                     save_root: str = os.path.join(RESULT_ROOT_BASE, "network"),
                     epoch: int = 0
                     ):
        """
        ネットワークのパラメータをセーブする
        :param save_root: セーブするディレクトリのパス
        :param epoch: エポック数
        :return: なし
        """
        save_dir = os.path.join(save_root, str(epoch))
        create_dir.create(save_dir)
        self.__generator.save(os.path.join(save_dir, "generator_iter.h5"))
        self.__discriminator.save(os.path.join(save_dir, "discriminator_.h5"))
        self.__combined.save(os.path.join(save_dir, "combined.h5"))


