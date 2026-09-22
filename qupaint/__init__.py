"""QuPAINT — inference package for the CVPR 2026 (Findings) model."""

from .model import QuPAINTConfig, QuPAINTModel
from .postprocess import denorm_box, draw_boxes, parse_boxes, parse_conclusion_boxes
from .preprocess import build_transform, dynamic_preprocess, load_image

__version__ = "1.0.0"
__all__ = [
    "QuPAINTConfig",
    "QuPAINTModel",
    "load_image",
    "build_transform",
    "dynamic_preprocess",
    "parse_boxes",
    "parse_conclusion_boxes",
    "denorm_box",
    "draw_boxes",
]
