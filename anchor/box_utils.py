from __future__ import division
import tensorflow as tf


def normalized_coordinates(proposal_boxes, image_height, image_width, scope='normalized_coordinates'):
    with tf.variable_scope(scope, 'normalized_coordinates', [proposal_boxes, image_width, image_height]):
        image_width = tf.cast(image_width, tf.float32)
        image_height = tf.cast(image_width, tf.float32)
        ymin, xmin, ymax, xmax = tf.split(proposal_boxes, num_or_size_splits=4, axis=1)
        normalized_ymin = ymin / (image_height - 1)
        normalized_xmin = xmin / (image_width - 1)
        normalized_ymax = ymax / (image_height - 1)
        normalized_xmax = xmax / (image_width - 1)
        return tf.concat([normalized_ymin, normalized_xmin, normalized_ymax, normalized_xmax], axis=1)


def iou(boxes1, boxes2, scope=None):
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxes1, boxes2)
        areas1 = area(boxes1)
        areas2 = area(boxes2)
        unions = (
            tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))


def area(anchors, scope=None):
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.unstack(tf.transpose(anchors))
        return (y_max - y_min) * (x_max - x_min)


def intersection(boxes1, boxes2, scope=None):
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxes1, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxes2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths
