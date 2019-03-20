# -*- coding: utf-8 -*-
from typing import Union
from typing import Tuple
from typing import Callable


def build_square_type(base_type: type):
    """
    任意の同じ型1変数か2組のタプルのアノテーションを作成する
    :param base_type: 基本となる型
    :return: アノテーションされる型
    """
    return Union[base_type, Tuple[base_type, base_type]]


raw_cal = build_square_type(int)


def init_get_pair(base_type: type):
    annotate_type = build_square_type(base_type)

    def get_pair(value: annotate_type)->Tuple[base_type, base_type]:
        return (value, value) if type(value) is base_type else value
    return get_pair


def init_pair_type(base_type: type):
    return build_square_type(base_type), init_get_pair(base_type)

