import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import config


def predict(image_features, is_training, scope='box_classifier'):
    with tf.variable_scope(scope, 'box_classifier', [image_features]):
        spatial_averaged_image_features = tf.reduce_mean(image_features, [1, 2],
                                                         keep_dims=True,
                                                         name='AvgPool')
        flattened_image_features = slim.flatten(spatial_averaged_image_features)
        if config.box_classifier_use_dropout:
            flattened_image_features = slim.dropout(flattened_image_features,
                                                    keep_prob=config.dropout_keep_prob,
                                                    is_training=is_training)

        box_encodings = slim.fully_connected(
            flattened_image_features,
            config.num_classes * 4,
            activation_fn=None,
            scope='BoxEncodingPredictor')
        scores = slim.fully_connected(
            flattened_image_features,
            config.num_classes + 1,
            activation_fn=None,
            scope='ClassPredictor')
        box_encodings = tf.reshape(box_encodings, [-1, config.num_classes, 4])
        scores = tf.reshape(scores, [-1, config.num_classes + 1])
        return box_encodings, scores
