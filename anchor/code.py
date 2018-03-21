import tensorflow as tf


def anchor_decode(anchors, rel_codes):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)

    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))

    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def get_center_coordinates_and_sizes(anchors):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(anchors))
    xcenter = (xmin + ymin) / 2
    ycenter = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return ycenter, xcenter, h, w


def anchor_encode(boxes, anchors):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ycenter, xcenter, h, w = get_center_coordinates_and_sizes(boxes)
    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    return tf.transpose(tf.stack([ty, tx, th, tw]))
