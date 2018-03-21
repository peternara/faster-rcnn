# coding=utf-8
import tensorflow as tf

from anchor import box_utils
from config import config
import sample


def anchor_assign(boxes, ground_truth_boxes, positive_thresold=0.7, negative_thresold=0.5, is_sample=True, scope=None):
    with tf.name_scope(scope, 'anchor_assign'):
        # 1.计算 gt_boxes 与 anchors的 iou矩阵
        iou_matrix = box_utils.iou(ground_truth_boxes, boxes)
        match_indices = tf.arg_max(iou_matrix, 0)  # 每个anchor对应的gt_box
        match_values = tf.reduce_max(iou_matrix, 0)

        # 2.计算正负样本的anchor索引
        positive_indices = tf.where(tf.greater_equal(match_values, positive_thresold))[:, 0]
        negative_indices = tf.where(tf.less(match_values, negative_thresold))[:, 0]

        # 3.正负样本采样
        if is_sample:
            positive_indices, negative_indices = sample.sample_anchors(positive_indices, negative_indices)

        matched_ground_truth_indices = tf.gather(match_indices, positive_indices)
        return positive_indices, negative_indices, matched_ground_truth_indices
