# -*- coding: utf-8 -*-
import importlib


def import_gan(will_use_tpu=True):
    """
    インポートする
    :param will_use_tpu: TPUに対応するものを呼び出すかどうか
    :return:
    """
    if will_use_tpu:
        return (importlib.import_module("Gan.tpu.gan"),
                importlib.import_module("Gan.tpu.generator"),
                importlib.import_module("Gan.tpu.discriminator")
                )
    else:
        return (importlib.import_module("Gan.normal.gan"),
                importlib.import_module("Gan.normal.generator"),
                importlib.import_module("Gan.normal.discriminator")
                )
