# coding=utf-8
import tensorflow as tf

from anchor import anchor_assigner, code
from config import config


def build_classifier_target(refined_box_encodings, predict_obeject_classes, proposal_boxes, ground_truth_boxes,
                            ground_truth_cls, scope=None):
    with tf.name_scope(scope, 'build_classifier_target'):
        # 1. 找出正负样本，并采样
        matched_anchor_indices, unmatched_anchor_indices, matched_ground_truth_indices = anchor_assigner.anchor_assign(
            proposal_boxes, ground_truth_boxes, positive_thresold=0.5, negative_thresold=0.5, is_sample=False)

        # 2. 分类目标
        mathed_cls_predict = tf.gather(predict_obeject_classes, matched_anchor_indices)
        unmatched_cls_predict = tf.gather(predict_obeject_classes, unmatched_anchor_indices)

        matchted_cls_target = tf.gather(ground_truth_cls, matched_ground_truth_indices)
        matchted_cls_target = tf.one_hot(matchted_cls_target, config.num_classes + 1)
        unmatched_cls_target = tf.tile([0], tf.size(unmatched_cls_predict))
        unmatched_cls_target = tf.one_hot(unmatched_cls_target, config.num_classes + 1)

        cls_predict = tf.concat([mathed_cls_predict, unmatched_cls_predict], axis=0)
        cls_target = tf.concat([matchted_cls_target, unmatched_cls_target], axis=0)

        # 3. 回归目标 待完成。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        # reg_predict = tf.gather(refined_box_encodings,
        #                         tf.concat([matched_anchor_indices, unmatched_anchor_indices], axis=0))
        # matched_anchors = tf.gather(anchors, matched_anchor_indices)
        # matched_gt_boxes = tf.gather(ground_truth_boxes, matched_ground_truth_indices)
        # matched_reg_target = code.anchor_encode(matched_gt_boxes, matched_anchors)
        #
        # unmatched_reg_target = tf.tile([0, 0, 0, 0], tf.size(unmatched_anchor_indices))
        # reg_target = tf.concat([matched_reg_target, unmatched_reg_target], axis=0)
        return cls_predict, cls_target, reg_predict, reg_target
    pass


def build_rpn_target(rpn_boxes_encodings, rpn_scores, anchors, ground_truth_boxes, scope=None):
    with tf.name_scope(scope, 'build_rpn_target'):
        # 1. 找出正负样本，并采样
        matched_anchor_indices, unmatched_anchor_indices, matched_ground_truth_indices = anchor_assigner.anchor_assign(
            anchors, ground_truth_boxes)

        # 2. 分类目标
        mathed_cls_predict = tf.gather(rpn_scores, matched_anchor_indices)
        unmatched_cls_predict = tf.gather(rpn_scores, unmatched_anchor_indices)
        cls_predict = tf.concat([mathed_cls_predict, unmatched_cls_predict], axis=0)

        matchted_cls_target = tf.tile([0, 1], tf.size(matched_anchor_indices))
        unmatched_cls_target = tf.tile([1, 0], tf.size(unmatched_anchor_indices))
        cls_target = tf.concat([matchted_cls_target, unmatched_cls_target], axis=0)

        # 3. 回归目标
        reg_predict = tf.gather(rpn_boxes_encodings,
                                tf.concat([matched_anchor_indices, unmatched_anchor_indices], axis=0))
        matched_anchors = tf.gather(anchors, matched_anchor_indices)
        matched_gt_boxes = tf.gather(ground_truth_boxes, matched_ground_truth_indices)
        matched_reg_target = code.anchor_encode(matched_gt_boxes, matched_anchors)

        unmatched_reg_target = tf.tile([0, 0, 0, 0], tf.size(unmatched_anchor_indices))
        reg_target = tf.concat([matched_reg_target, unmatched_reg_target], axis=0)
        return cls_predict, cls_target, reg_predict, reg_target
