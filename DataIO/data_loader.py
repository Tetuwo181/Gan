# -*- coding: utf-8 -*-
from keras.utils import np_utils
import os
import cv2
import numpy as np
from typing import List
from functools import reduce
from operator import add
from util_types import two_dim
from typing import Optional
from numba import jit

img_size, size_converter = two_dim.init_pair_type(int)


def load_dataset(root_dir: str, is_normalize: bool = True, img_resize_val: Optional[img_size] = None):
    """
    画像データを読み込む
    :param root_dir: 画像データの格納されているルートディレクトリ。直下に存在するディレクトリ名が各画像のクラス名に
    :param is_normalize: 正規化したデータを返すかどうか
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :return: numpy形式の画像データの配列とラベルの配列のタプル
    """
    class_names = [path for path in os.listdir(root_dir)]
    print("all classes", class_names)
    encoder = label_encoder(class_names)
    result_img_set = []
    result_label_set = []
    for class_name in class_names:
        got_data = load_data_in_dir(os.path.join(root_dir, class_name), is_normalize, img_resize_val)
        label_converted = [encoder(class_name) for index in range(len(got_data))]
        result_img_set.extend(got_data)
        result_label_set.extend(label_converted)
        print("class", class_name, "loaded data_num", len(got_data))
    return np.array(result_img_set), np.array(result_label_set)


def load_data_in_dir(dir_path: str, is_normalize: bool = True, img_resize_val: Optional[img_size] = None):
    """
    指定したディレクトリに存在するデータを読み込む
    :param dir_path: 画像データの格納されているディレクトリ。
    :param is_normalize: 正規化したデータを返すかどうか
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :return: numpy形式の画像データの配列
    """
    print("load", dir_path)
    img_path_set = [os.path.join(dir_path,  data_name) for data_name in os.listdir(dir_path)]
    img_set = [load_img(img_path, img_resize_val) for img_path in img_path_set]
    return normalise_img_set(np.array(img_set)) if is_normalize else np.array(img_set)


@jit
def load_img(img_path: str, img_resize_val: Optional[img_size] = None):
    """
    指定されたパスの画像ファイルを読み込む
    :param img_path: 画像ファイル
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :return:
    """
    raw_img = cv2.imread(img_path)
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    if img_resize_val is None:
        return img
    resize_val = size_converter(img_resize_val)
    return cv2.resize(img, resize_val)


def label_encoder(class_set: List[str]):
    """
    ベースとなるクラスのリストを与えてエンコードする関数を返す
    :param class_set: ベースとなるクラスのリスト
    :return: keras形式でエンコードされたラベルのインデックスを返す関数
    """
    @jit
    def encode(base_class: str):
        """
        クラスとなる文字列を与えてkeras形式でエンコードされたクラスのインデックスを返す
        :param base_class: ベースとなるクラス名
        :return: エンコードされたクラス名
        """
        label_encoded = class_set.index(base_class)
        return np_utils.to_categorical(label_encoded, len(class_set))
    return encode


@jit
def normalise_img(img: np.ndarray)->np.ndarray:
    """
    画像を正規化
    :param img: 正規化する対象
    :return: 正規化後の配列
    """
    return (img.astype(np.float32) - 127.5) / 127.5


def normalise_img_set(img_set: np.ndarray)->np.ndarray:
    """
    画像を入力するために正規化
    :param img_set: ベースになるデータ
    :return: 正規化後のデータ
    """
    return np.array([normalise_img(img) for img in img_set])


@jit
def sampling_real_data_set(batch_num: int, img_set: np.ndarray)->np.ndarray:
    """
    実際のデータセットからbatch_num分だけデータを復元抽出する
    :param batch_num: データを抽出する数
    :param img_set: 元となるデータセット
    :return: 抽出されたデータセット
     """
    chosen_id_set = np.random.randint(0, img_set.shape[0], batch_num)
    return img_set[chosen_id_set]
