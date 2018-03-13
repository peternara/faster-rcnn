import tensorflow as tf
from anchor import anchor_generate
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    sess = tf.Session()

    height = 10
    width = 6
    scales = [0.5, 1, 2]
    aspect_ratio = [0.5, 1, 2]
    base_anchor_size = 16
    anchor_stride = 16
    anchors = anchor_generate.anchor_generate(height, width, scales, aspect_ratio, base_anchor_size, anchor_stride)
    sess.run(tf.global_variables_initializer())
    output = sess.run(anchors)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.xlim(-50,200)
    plt.ylim(-50,200)
    ax1.add_patch(
        patches.Rectangle(
            (0, 0),  # (x,y)
            96,  # width
            160,  # height

        )
    )
    for anchor in output[:9]:
        xmin, ymin, xmax, ymax = anchor
        ax1.add_patch(
            patches.Rectangle(
                (xmin, ymin),  # (x,y)
                xmax - xmin,  # width
                ymax - ymin,  # height
                fill=False
            )
        )
        print((ymax-ymin)*(xmax-xmin))

    fig1.show()
