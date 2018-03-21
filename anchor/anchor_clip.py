import tensorflow as tf

from anchor.box_utils import area


def anchor_clip(anchors, window, scope='ClipToWindow'):
    with tf.name_scope(scope, 'ClipToWindow'):
        y_min, x_min, y_max, x_max = tf.split(value=anchors, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
        y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
        x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
        x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
        clipped = tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1)

        areas = area(clipped)
        nonzero_area_indices = tf.cast(tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
    return tf.gather(clipped, nonzero_area_indices), nonzero_area_indices


def prune_outside_window(anchors, window, scope=None):
    with tf.name_scope(scope, 'PruneOutsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(value=anchors, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.less(y_min, win_y_min), tf.less(x_min, win_x_min),
            tf.greater(y_max, win_y_max), tf.greater(x_max, win_x_max)
        ], 1)
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return valid_indices
