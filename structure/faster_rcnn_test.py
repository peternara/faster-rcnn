import tensorflow as tf
from anchor import box_utils
from structure import faster_rcnn

if __name__ == '__main__':
    with tf.Session() as sess:
        image = tf.random_normal([1, 900, 800, 3])
        gt_boxes = tf.constant([[0, 0, 700, 600],
                                [40, 50, 100, 200]], dtype=tf.float32)
        image_shape = tf.shape(image)
        height, width = image_shape[1], image_shape[2]
        normalized_gt_boxes = tf.cast(box_utils.normalized_coordinates(gt_boxes, height, width), tf.float32)

        predict_dict = faster_rcnn.model(image, normalized_gt_boxes, True)
        sess.run(tf.global_variables_initializer())
        print(normalized_gt_boxes.eval())
