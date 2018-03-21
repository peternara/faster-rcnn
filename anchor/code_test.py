import tensorflow as tf
import anchor.code as decode


class DecodeTest(tf.test.TestCase):
    def testDecode(self):
        with self.test_session() as sess:
            anchors = tf.Variable([[0, 0, 10, 10]], dtype=tf.float32)
            relcode = tf.Variable([[.1, .2, .3, .4]], dtype=tf.float32)
            decoded_anchors = decode.anchor_decode(anchors,relcode)
            sess.run(tf.global_variables_initializer())
            output = sess.run(decoded_anchors)
            print(output)