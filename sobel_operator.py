import math

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


# TODO - implement gx and gy standalone methods
def compute_gradient_magnitude(image):
    # this will store our gradient magnitude results
    gradient = []

    padding = int(len(gx_mask)/2)

    # iterate over the original image
    for i in range(padding, len(image) - padding):
        for j in range(padding, len(image[i]) - padding):
            # horizontal direction
            xsum = 0
            for k in range(len(gx_mask)):
                for l in range(len(gx_mask[i])):
                    xshift = k - 1
                    yshift = l - 1
                    xsum += image[i + xshift][j + yshift] * gx_mask[k][l]

            # vertical direction
            ysum = 0
            for m in range(len(gy_mask)):
                for n in range(len(gy_mask[m])):
                    xshift = m - 1
                    yshift = n - 1
                    ysum += image[i + xshift][j + yshift] * gy_mask[m][n]

            # compute gradient magnitude
            gradient[i][j] = math.sqrt((xsum * xsum) + (ysum * ysum))

    return gradient
