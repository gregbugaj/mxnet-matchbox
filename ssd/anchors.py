import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
import matplotlib.pyplot as plt


def main():
    n = 40
    # shape: batch x channel x height x weight
    x = nd.random_uniform(shape=(1, 3, n, n))

    y = MultiBoxPrior(x, sizes=[.5, .25, .1], ratios=[1, 2, .5])

    print(y)
    # the first anchor box generated for pixel at (20,20)
    # its format is (x_min, y_min, x_max, y_max)
    boxes = y.reshape((n, n, -1, 4))
    print('The first anchor box at row 21, column 21:', boxes[20, 20, 0, :])

    colors = ['blue', 'green', 'red', 'black', 'magenta']
    plt.imshow(nd.ones((n, n, 3)).asnumpy())
    anchors = boxes[20, 20, :, :]

    for i in range(anchors.shape[0]):
        plt.gca().add_patch(box_to_rect(anchors[i,:]*n, colors[i]))
    plt.show()


def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]),
        fill=False, edgecolor=color, linewidth=linewidth)

if __name__ == "__main__":
    main()
