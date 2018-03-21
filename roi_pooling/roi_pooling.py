import tensorflow as tf
import tensorflow.contrib.slim as slim


def pooling(feature_maps, normalized_proposal_boxes, scope='roi_pooling'):
    with tf.variable_scope(scope, 'roi_pooling', [feature_maps, normalized_proposal_boxes]):
        box_ind = tf.zeros([tf.shape(normalized_proposal_boxes)[0]], dtype=tf.int32)
        cropped_regions = tf.image.crop_and_resize(feature_maps, normalized_proposal_boxes, box_ind, [14, 14])
        return slim.max_pool2d(cropped_regions, [2, 2])
