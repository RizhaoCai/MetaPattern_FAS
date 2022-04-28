"""
  MetaTexFPNv6_2 is based on MetaTexFPNv6_1.
  Difference: it uses the conv with kernel=3

"""
from .trainer import Trainer
from .custom_config import _C as custom_cfg


__all__ = [
    'Trainer',
    'custom_cfg'
]

