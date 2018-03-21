# coding=utf-8
import tensorflow as tf
from losss.build_target import build_classifier_target, build_rpn_target


def smooth_l1_loss(reg_predict, reg_target):
    diff = reg_predict - reg_target
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), 1)

    return tf.reduce_mean(anchorwise_smooth_l1norm)


def rpn_loss(rpn_boxes_encodings, rpn_scores, anchors, ground_truth_boxes, scope=None):
    with tf.name_scope(scope, 'rpn_loss', [rpn_boxes_encodings, rpn_scores, anchors, ground_truth_boxes]):
        cls_predict, cls_target, reg_predict, reg_target = build_rpn_target(rpn_boxes_encodings, rpn_scores, anchors,
                                                                            ground_truth_boxes)

        # 分类损失
        objectness_losses = tf.nn.softmax_cross_entropy_with_logits(labels=cls_target, logits=cls_predict)
        objectness_losses = tf.reduce_mean(objectness_losses)

        # 6.位置损失
        localization_losses = smooth_l1_loss(reg_predict, reg_target)

        return objectness_losses, localization_losses


def cls_loss(refined_box_encodings, predict_obeject_classes, proposal_boxes, ground_truth_boxes, ground_truth_cls):
    with tf.name_scope('BoxClassifierLoss'):
        cls_predict, cls_target, reg_predict, reg_target = build_classifier_target(refined_box_encodings,
                                                                                   predict_obeject_classes,
                                                                                   proposal_boxes,
                                                                                   ground_truth_boxes,
                                                                                   ground_truth_cls)

        # 分类损失
        objectness_losses = tf.nn.softmax_cross_entropy_with_logits(labels=cls_target, logits=cls_predict)
        objectness_losses = tf.reduce_mean(objectness_losses)

        # 6.位置损失
        localization_losses = smooth_l1_loss(reg_predict, reg_target)

        return objectness_losses, localization_losses


def to_absolute_coordinates(ground_truth_boxes, image_shape):
    height, width = image_shape[0], image_shape[1]
    ymin, xmin, ymax, xmax = tf.split(ground_truth_boxes, num_or_size_splits=4, axis=1)

    ymin = ymin * (height - 1)
    xmin = xmin * (width - 1)
    ymax = ymax * (height - 1)
    xmax = xmax * (width - 1)

    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def total_loss(predict_dict, ground_truth_boxes, ground_truth_cls, scope=None):
    with tf.name_scope(scope, 'total_loss', predict_dict.values()):
        ground_truth_boxes = to_absolute_coordinates(ground_truth_boxes, predict_dict['image_shape'])

        rpn_objectness_loss, rpn_localization_loss = rpn_loss(predict_dict['rpn_box_encoding'],
                                                              predict_dict['rpn_scores'],
                                                              predict_dict['anchors'],
                                                              ground_truth_boxes)
        classifier_objectness_loss, classifier_localization_loss = cls_loss(predict_dict['refined_box_encodings'],
                                                                            predict_dict['obeject_classes'],
                                                                            predict_dict['proposal_boxes'],
                                                                            ground_truth_boxes,
                                                                            ground_truth_cls)

        total = rpn_objectness_loss + rpn_localization_loss + classifier_objectness_loss + classifier_localization_loss
        return rpn_localization_loss, rpn_objectness_loss, classifier_localization_loss, classifier_objectness_loss, total
