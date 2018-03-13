# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim

from anchor import anchor_generate, anchor_clip
from config import config
from region_proposals import sample


def rpn(inputs,
        anchor_nums=9,
        ground_truth_boxes=None,
        is_training=True,
        scope='rpn'):
    """
    推荐框生成网络
    :param inputs: 输入一个特征图 [1,width,width,channels]
    :param is_training: 是否训练
    :param scope: 变量作用域
    :return: end_points: a dict of tensors with intermediate activations.
    """

    with tf.variable_scope(scope, 'rpn', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            _, height, width, _ = inputs.get_shape().as_list()
            net = slim.conv2d(inputs, 256, [3, 3], scope='conv1')
            # rpn 分类分支
            scores = slim.conv2d(net, 2 * anchor_nums, [1, 1], scope='cls_conv1')
            scores = tf.reshape(scores, [-1, 2], name='cls_reshape')
            scores = slim.softmax(scores, scope='cls_softmax')

            # rpn anchor 预测分支
            anchor_encodes = slim.conv2d(net, 4 * anchor_nums, [1, 1], scope='reg_conv1', activation_fn=tf.nn.sigmoid)
            anchor_encodes = tf.reshape(anchor_encodes, [-1, 4])

            # 1.生成anchor
            anchors = anchor_generate.anchor_generate(height, width, config.scales, config.aspect_ratios,
                                                      config.base_anchor_size, config.anchor_stride)

            # 2.剪裁anchor
            clip_window = tf.to_float(tf.stack([0, 0, height * config.anchor_stride, width * config.anchor_stride]))
            if is_training:
                # 去掉超出图像的框
                keep = anchor_clip.prune_outside_window(anchors, clip_window)
                scores = tf.gather(scores, keep)
                anchors = tf.gather(anchors, keep)
                anchor_encodes = tf.gather(anchor_encodes, keep)
            else:
                anchors, keep = anchor_clip.anchor_clip(anchors, clip_window)
                scores = tf.gather(scores, keep)
                anchor_encodes = tf.gather(anchor_encodes, keep)

            prediction_dict = {
                'rpn_box_predictor_features': rpn_box_predictor_features,
                'rpn_features_to_crop': rpn_features_to_crop,
                'image_shape': image_shape,
                'rpn_box_encodings': rpn_box_encodings,
                'rpn_objectness_predictions_with_background':
                    rpn_objectness_predictions_with_background,
                'anchors': anchors
            }