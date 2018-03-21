from nets import vgg
import tensorflow as tf


def extract(images, extractor='vgg16', scope='feature_extract'):
    with tf.variable_scope(scope, 'feature_extract', [images]):
        if extractor == 'vgg16':
            return vgg.vgg_16(images)
