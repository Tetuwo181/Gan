# -*- coding: utf-8 -*-
import os
from numba import jit


@jit
def create(dir_path):
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)
