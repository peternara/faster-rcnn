import tensorflow as tf
from nets import vgg


class VGG16Test(tf.test.TestCase):
    def testForward(self):
        batch_size = 1
        height, width = 224, 224
        with self.test_session() as sess:
            inputs = tf.random_uniform((batch_size, height, width, 3))
            features = vgg.vgg_16(inputs)
            sess.run(tf.global_variables_initializer())
            output = sess.run(features)
            self.assertTrue(output.any())
