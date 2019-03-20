from keras import optimizers
from typing import Union
from util_types import two_dim

TypeOptimizer = Union[optimizers.Adam,
                      optimizers.SGD,
                      optimizers.RMSprop,
                      optimizers.Adagrad,
                      optimizers.Nadam,
                      optimizers.TFOptimizer]

type_optimizer_list = [optimizers.Adam,
                       optimizers.SGD,
                       optimizers.RMSprop,
                       optimizers.Adagrad,
                       optimizers.Nadam,
                       optimizers.TFOptimizer]

GanOptimizer = two_dim.build_square_type(TypeOptimizer)
input_img_size, get_size_pair = two_dim.init_pair_type(int)


def get_optimizer_set(val_optimizer):
    if(type(val_optimizer)) is tuple:
        return val_optimizer
    return two_dim.init_get_pair(type(val_optimizer))(val_optimizer)
