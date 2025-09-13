"""
Matrix subpackage containing functions neccesary to transform a matrix into reduced Row Echelon Form
"""

from .elementary import rowswap 
from .elementary import rowscale
from .elementary import rref


__all__ = ['rowswap', 'rowscale', 'rref']
