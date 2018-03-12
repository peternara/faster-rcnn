# coding=utf-8
import config.config as config
from anchor import anchor_generate
from anchor import anchor_clip
from anchor import decode
import tensorflow as tf


def anchor_sample(cls, reg_anchors, height, width, ground_truth_boxes=None, is_training=True, scope='rpn_sample'):
    """
    对rpn网络生成的框进行采样
    :param is_training:
    :param stride: VGG16的步长 16
    :param cls: anchor分数 [-1,2]
    :param reg_anchor: anchor预测坐标[tx,ty,tw,th],shape:[-1,4]
    :param ground_truth_boxes:真实框坐标 [x1,y1,x2,y2],相对w,h的相对坐标
    :param scope:变量范围
    :return:
    """
    # 1.生成anchor
    base_anchors = anchor_generate.anchor_generate(height, width, config.scales, config.aspect_ratios,
                                                   config.base_anchor_size, config.anchor_stride)

    # 2. anchor 回归
    anchors = decode.anchor_decode(base_anchors, reg_anchors)

    # 3.剪裁anchor
    keep = anchor_clip.anchor_clip(anchors, height, width, config.anchor_stride, is_training)

    pass
