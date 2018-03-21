# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim

from anchor import anchor_generate
from config import config


def region_proposals_network(feature_maps, scope):
    with tf.variable_scope(scope, 'region_proposals_network', [feature_maps]):
        anchor_nums = len(config.aspect_ratios) * len(config.scales)

        net = slim.conv2d(feature_maps, 256, [3, 3], scope='conv1')
        # rpn 分类
        scores = slim.conv2d(net, 2 * anchor_nums, [1, 1], scope='cls_conv1')
        scores = tf.reshape(scores, [-1, 2], name='cls_reshape')
        # scores = slim.softmax(scores, scope='cls_softmax')

        # rpn anchor 预测回归
        anchor_encodes = slim.conv2d(net, 4 * anchor_nums, [1, 1], scope='reg_conv1', activation_fn=tf.nn.sigmoid)
        anchor_encodes = tf.reshape(anchor_encodes, [-1, 4])

        # 生成anchors
        feature_shape = tf.shape(feature_maps)
        feature_height = feature_shape[1]
        feature_width = feature_shape[2]
        anchors = anchor_generate.anchor_generate(feature_height, feature_width, config.scales, config.aspect_ratios,
                                                  config.base_anchor_size, config.anchor_stride)
        return scores, anchor_encodes, anchors
