import math
from scipy.special import gamma
import numpy
import cv2


def energy_calc_1D(im):
    """
    1D energy calculating for potts model.
    :param im: Image for which labels to be calculated
    :return: Label matrix of image
    """
    part_labels = numpy.zeros((512, 512))
    for x in range(1, 511):
        for y in range(1, 511):
            if (im[x, y - 1] - 5 < im[x, y] < im[x, y - 1] + 5) and \
                    (im[x, y + 1] - 5 < im[x, y] < im[x, y + 1] + 5):
                im[x, y] = im[x, y - 1]
                part_labels[x, y] = 0
            elif (im[x - 1, y] - 40 < im[x, y] < im[x - 1, y] + 40) and \
                    (im[x + 1, y] - 40 < im[x, y] < im[x + 1, y] + 40):
                im[x, y] = im[x - 1, y]
                part_labels[x, y] = 1
    return im, part_labels


def energy_calc_2D(im):
    """
    2D energy calculating for potts model.
    :param im: Image for which labels to be calculated
    :return: Label matrix of image
    """
    part_labels = numpy.zeros((512, 512))
    for x in range(1, 511):
        for y in range(1, 511):
            if (im[x - 1, y - 1] - 5 < im[x, y] < im[x + 1, y - 1] + 5) and \
                    (im[x - 1, y] - 5 < im[x, y] < im[x + 1, y] + 5) and\
                    (im[x - 1, y + 1] - 5 < im[x, y] < im[x + 1, y + 1] + 5):
                im[x, y] = im[x, y - 1]
                part_labels[x, y] = 0
            elif (im[x - 1, y - 1] - 5 < im[x, y] < im[x + 1, y - 1] + 5) and \
                    (im[x - 1, y] - 5 < im[x, y] < im[x + 1, y] + 5) and\
                    (im[x - 1, y + 1] - 5 < im[x, y] < im[x + 1, y + 1] + 5):
                im[x, y] = im[x - 1, y]
                part_labels[x, y] = 1
    return part_labels


def translate(value, leftMin, leftMax, rightMin, rightMax):
    """
    Normalize the data in range rightMin and rightMax
    :param value: Value to be normalize
    :param leftMin: original min value
    :param leftMax: original max value
    :param rightMin: final min value
    :param rightMax: final max value
    :return: Normalized value
    """
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


def l_norm(x, l):
    """
    L-Norm for vector x
    :param x: vector x
    :param l: norm
    :return: l normed value of vector x
    """
    lnorm = 0
    for i in x:
        lnorm += numpy.abs(i) ** p
    lnorm ** (1. / l)
    return lnorm


def param_estimator(image):
    """
    Calculate assumed shape and scale vector from images
    :param image: image
    :return: assumed shape, scale vector and divided matrix
    """
    shape_vector_assumed = []
    scale_vector_assumed = []
    mat = cv2.imread(image, 0)
    mat = cv2.resize(mat, (512, 512))
    sub_mat = [
        mat[0:128, 0:128],
        mat[0:128, 128:256],
        mat[0:128, 256:384],
        mat[0:128, 384:513],
        mat[128:256, 0:128],
        mat[128:256, 128:256],
        mat[128:256, 256:384],
        mat[128:256, 384:513],
        mat[256:384, 0:128],
        mat[256:384, 128:256],
        mat[256:384, 256:384],
        mat[256:384, 384:513],
        mat[384:513, 0:128],
        mat[384:513, 128:256],
        mat[384:513, 256:384],
        mat[384:513, 384:513]
    ]
    for i in sub_mat:
        shape_vector_assumed.append(translate(
            value=numpy.std(numpy.reshape(i, 128 * 128)),
            leftMin=numpy.min(numpy.reshape(i, 128 * 128)),
            leftMax=numpy.max(numpy.reshape(i, 128 * 128)),
            rightMin=0, rightMax=3
        ))
        scale_vector_assumed.append(translate(
            value=numpy.mean(numpy.reshape(i, 128 * 128)),
            leftMin=numpy.min(numpy.reshape(i, 128 * 128)),
            leftMax=numpy.max(numpy.reshape(i, 128 * 128)),
            rightMin=0, rightMax=3
        ))
    return shape_vector_assumed, scale_vector_assumed, sub_mat


def trf_prior(im):
    """
    Generate TRF priors
    :param im: Image
    :return: Prior value of TRF
    """
    ans = []
    sh_v, sc_v, sub_mat = param_estimator(im)
    for i in range(len(sub_mat)):
        ans.append((1 / pow((2 * pow(sc_v[i], 1/sh_v[i]) * gamma(1 + (1 / sh_v[i]))), sub_mat[i].size)) *
                    (math.exp(-l_norm(numpy.reshape(sub_mat[i]), sh_v[i]) / sc_v[i])))
    return numpy.prod(ans)


