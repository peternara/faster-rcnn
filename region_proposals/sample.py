# coding=utf-8
import config.config as config
from anchor import anchor_generate
from anchor import anchor_clip
from anchor import decode
import tensorflow as tf


def anchor_sample(scores, anchor_encodes, height, width, ground_truth_boxes=None, is_training=True, scope='rpn_sample'):
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


    # 4.nms处理
    selected_indices = tf.image.non_max_suppression(
        anchors, cls[:, 0],
        config.max_output_size, iou_threshold=config.iou_threshold)

    anchors = tf.gather(anchors, selected_indices)
    cls = tf.gather(cls, selected_indices)

    # 正负采样
    ...
    return anchors, cls
