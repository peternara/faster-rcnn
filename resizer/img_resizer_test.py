import tensorflow as tf
from resizer import img_resizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with tf.Session() as sess:
        image_raw_data_jpg = tf.gfile.FastGFile('/Users/wang/Pictures/IMG_20170222_140239.jpg', 'rb').read()
        img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
        img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
        img_data_jpg = tf.expand_dims(img_data_jpg, axis=0)
        img = img_resizer.resizer(img_data_jpg, 400)

        sess.run(tf.global_variables_initializer())

        print(tf.shape(img_data_jpg).eval())
        print(tf.shape(img).eval())

        plt.figure(0)
        plt.imshow(tf.squeeze(img_data_jpg, axis=0).eval())
        plt.figure(1)
        plt.imshow(tf.squeeze(img, axis=0).eval())
        plt.show()
