# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sample


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
            cls = slim.conv2d(net, 2 * anchor_nums, [1, 1], scope='cls_conv1')
            cls = tf.reshape(cls, [-1, 2], name='cls_reshape')
            cls = slim.softmax(cls, scope='cls_softmax')

            # rpn anchor 预测分支
            anchors = slim.conv2d(net, 4 * anchor_nums, [1, 1], scope='reg_conv1', activation_fn=tf.nn.sigmoid)
            anchors = tf.reshape(anchors, [-1, 4])

            sampled_cls, sampled_anchors = sample.anchor_sample(cls, anchors, height, width, ground_truth_boxes,
                                                                is_training, scope='rpn_sample')
            return sampled_cls, sampled_anchors
