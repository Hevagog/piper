from enum import Enum


class AugmentationType(Enum):
    FLIP = "flip"
    ROTATE = "rotate"
    COLOR_JITTER = "color_jitter"
    RANDOM_CROP = "random_crop"
    CENTER_CROP = "center_crop"
