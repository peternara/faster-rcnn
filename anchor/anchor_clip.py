import tensorflow as tf


def anchor_clip(anchors, height, width, anchor_stride, is_training=True):
    ymax = height * anchor_stride
    xmax = width * anchor_stride

    pass
