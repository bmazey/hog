import math
import numpy


gx_mask = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

gy_mask = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
]


def compute_horizontal_gradient_magnitude(image):
    gradient = [[0] * len(image[0]) for _ in range(len(image))]
    padding = int(len(gx_mask) / 2)

    # iterate over the original image
    for i in range(padding, len(image) - padding):
        for j in range(padding, len(image[i]) - padding):
            # horizontal direction
            xsum = 0
            for k in range(len(gx_mask)):
                for l in range(len(gx_mask[k])):
                    xshift = k - 1
                    yshift = l - 1
                    # take first position because 3d matrix
                    xsum += image[i + xshift][j + yshift][0] * gx_mask[k][l]

            # normalize
            gradient[i][j] = xsum / 4

    return gradient


def compute_vertical_gradient_magnitude(image):
    gradient = [[0] * len(image[0]) for _ in range(len(image))]
    padding = int(len(gy_mask) / 2)

    # iterate over the original image
    for i in range(padding, len(image) - padding):
        for j in range(padding, len(image[i]) - padding):
            # vertical direction
            ysum = 0
            for m in range(len(gy_mask)):
                for n in range(len(gy_mask[m])):
                    xshift = m - 1
                    yshift = n - 1
                    # take first position because 3d matrix
                    ysum += image[i + xshift][j + yshift][0] * gy_mask[m][n]

            # normalize
            gradient[i][j] = ysum / 4

    return gradient


def compute_gradient_magnitude(gx_image, gy_image):
    # sanity check
    assert len(gx_image) == len(gy_image)
    assert len(gx_image[0]) == len(gy_image[0])

    # this will store our gradient magnitude results
    gradient = [[0] * len(gx_image[0]) for _ in range(len(gx_image))]
    padding = int(len(gx_mask)/2)

    # iterate over the original image
    for i in range(padding, len(gx_image) - padding):
        for j in range(padding, len(gx_image[i]) - padding):

            # compute magnitude
            gradient[i][j] = numpy.round(math.sqrt((gx_image[i][j] * gx_image[i][j])
                                                   + (gy_image[i][j] * gy_image[i][j])))

    return gradient


# TODO - compute gx and gy separately
def compute_gradient_angle(gradient, gx, gy):
    # create new array to store result
    angle = [[0] * len(gradient[0]) for _ in range(len(gradient))]

    for i in range(len(gradient)):
        for j in range(gradient[i]):
            if gradient[i][j] == 0:
                angle[i][j] = 0
            else:
                angle[i][j] = numpy.arctan2(gy[i][j], gx[i][j])
