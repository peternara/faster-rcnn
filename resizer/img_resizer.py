import tensorflow as tf
from config import config


def resizer(images, resize_shorter_edge_size, scope='resize_image'):
    with tf.variable_scope(scope, 'resize_image', [images]):
        shape = tf.shape(images)
        height, width = shape[1], shape[2]
        shoter_edge_size = tf.constant(resize_shorter_edge_size, dtype=tf.int32)
        height_smaller_than_width = tf.less_equal(height, width)
        new_height, new_width = tf.cond(
            height_smaller_than_width,
            lambda: (shoter_edge_size, tf.cast(shoter_edge_size * width / height, tf.int32)),
            lambda: (tf.cast(shoter_edge_size * height / width, tf.int32), shoter_edge_size)
        )

        return tf.image.resize_images(images, [new_height, new_width])
