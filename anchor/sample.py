# coding=utf-8
import config.config as config
import tensorflow as tf
from anchor import anchor_assigner


def sample_box_classifier_batch(proposal_boxes, scores, ground_truth_boxes):
    # 判断 每个proposal 对应的 gt_boxe
    mathed_anchor_indices, unmatched_anchor_indices, matched_ground_truth_indices = anchor_assigner.anchor_assign(
        proposal_boxes, ground_truth_boxes)

    indices = tf.concat([mathed_anchor_indices, unmatched_anchor_indices], axis=0)
    proposal_boxes = tf.gather(proposal_boxes, indices)
    scores = tf.gather(scores, indices)
    return proposal_boxes, scores


def sample(indices, size):
    indices = tf.random_shuffle(indices)
    size = tf.minimum(tf.size(indices), size)
    return indices[0:size]


def sample_anchors(positive_indices, negative_indices):
    positive_size = int(config.positive_fraction * config.batch_size)
    negative_size = config.batch_size - positive_size

    positive_indices = sample(positive_indices, positive_size)
    negative_indices = sample(negative_indices, negative_size)
    return positive_indices, negative_indices
