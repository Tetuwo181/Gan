from typing import Tuple
from collections import namedtuple

GanPair = namedtuple("GanPair", ("generator", "discriminator"))

CycleInput = Tuple[GanPair, GanPair]
