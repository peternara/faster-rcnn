import tensorflow as tf


def anchor_decode(anchors, rel_codes):
    xcenter_a, ycenter_a, wa, ha = get_center_coordinates_and_sizes(anchors)

    tx, ty, tw, th = tf.unstack(tf.transpose(rel_codes))

    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))


def get_center_coordinates_and_sizes(anchors):
    xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(anchors))
    xcenter = (xmin + ymin) / 2
    ycenter = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return xcenter, ycenter, w, h
