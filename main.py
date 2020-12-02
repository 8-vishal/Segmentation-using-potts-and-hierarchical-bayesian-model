import numpy
import math
import cv2
import random
import matplotlib
from scipy.special import gamma
import matplotlib.pyplot as plt
from functions import *
matplotlib.use("TkAgg")


def inverse_gamma(data, alpha=0.1, beta=0.1):
    """
    Inverse gamma distributions
    :param data: Data value
    :param alpha: alpha value
    :param beta: beta value
    :return: Inverse gamma distributiion
    """
    return (pow(beta, alpha) / math.gamma(alpha)) *\
           pow(alpha, data-1) * math.exp(-beta/data)


def shape_scale():
    """
    Generate shape and scale params
    :return:
    """
    pre_shape = numpy.random.uniform(0, 3, 16)
    pre_scale = numpy.random.uniform(0, 3, 16)
    shape = [inverse_gamma(i) for i in pre_shape]
    scale = [inverse_gamma(j) for j in pre_scale]
    return shape, scale


def noise_variance():
    """
    Noise varriance sampled in inverse gamma distribution
    :return: Sampled Noise
    """
    var_list = numpy.arange(0, 3.2, 0.2)
    var = inverse_gamma(random.choice(var_list))
    return numpy.random.normal(0, var, (512, 512))


def GGD(shape, scale, path):
    """
    Generalised gaussian distribution of input image
    :param shape: Shape param of GGD
    :param scale: Scale param of GGD
    :param path: Path to image
    :return: Approximate GGD image
    """
    def ggd(x):
        p1 = 2 * pow(scale, 1 / shape) * gamma(1 + 1 / shape)
        p2 = math.exp(-pow(abs(x), shape) / scale)
        return (1 / p1) * p2

    mat = cv2.imread(path, 0)
    mat = cv2.resize(mat, (512, 512))
    mata = numpy.zeros(mat.shape)
    for i in range(len(mat)):
        for j in range(len(mat)):
            mata[i][j] = ggd(mat[i][j])
    return mat, mata


def potts_label_map_trf(image):
    """
    Find Label Map for the ground truth image
    :param image: Path to Image
    :return: Approximate segmented and labels for segmentation
    """
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (512, 512))
    im = img.copy()
    '''labels = numpy.zeros((512, 512))
    for x in range(1, 511):
        for y in range(1, 511):
            if (im[x, y - 1] - 5 < im[x, y] < im[x, y - 1] + 5) and\
                    (im[x, y + 1] - 5 < im[x, y] < im[x, y + 1] + 5):
                im[x, y] = im[x, y - 1]
                labels[x, y] = 0
            elif (im[x - 1, y] - 40 < im[x, y] < im[x - 1, y] + 40) and\
                    (im[x + 1, y] - 40 < im[x, y] < im[x + 1, y] + 40):
                im[x, y] = im[x - 1, y]
                labels[x, y] = 1'''
    return energy_calc_1D(im)


def Segment(img):
    """
    Segment the image based on POTTS model
    :param img: Approximated Image
    :return: Segmented image in 4 classes
    """
    out = img.copy()
    for i in range(len(img)):
        for j in range(len(img)):
            if img[i, j] in range(0, 41):
                out[i, j] = 0
            elif img[i, j] in range(41, 100):
                out[i, j] = 1
            elif img[i, j] in range(150, 255):
                out[i, j] = 2
            else:
                out[i, j] = 3
    return out


def plot(path):
    orig, approx = GGD(0.1, 0.1, path)
    approx_seg, labels = potts_label_map_trf(path)
    final_segment = Segment(approx_seg)

    plt.figure(figsize=(20, 10))
    plt.legend()
    plt.subplot(221)
    plt.imshow(orig, cmap='gray'), plt.title("Original Image")
    plt.subplot(222)
    plt.imshow(approx, cmap='gray'), plt.title("GGD approximated Image")
    plt.subplot(223)
    plt.imshow(approx_seg, cmap='gray'), plt.title("TRF segmented Image")
    plt.subplot(224)
    plt.imshow(final_segment, cmap='jet'), plt.title("Final Segmented Image")
    plt.show()


if __name__ == "__main__":
    # To run the code just change the path and input path of the image.
    # For windows last line of code will be like -  plot(path=r".\us_images\18-19-54.jpg")
    # For linux last line of code will be like   -  plot(path="./us_images/18-19-54.jpg")
    plot(path="./us_images/18-19-54.jpg")