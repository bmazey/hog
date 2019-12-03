from image_converter import get_image_array, convert_grayscale
from sobel_operator import compute_gradient_magnitude, compute_horizontal_gradient_magnitude, \
    compute_vertical_gradient_magnitude, compute_gradient_angle
from histogram import Histogram


def detect():
    image = get_image_array('C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive\\crop_000010b.bmp')
    gx_gradient = compute_horizontal_gradient_magnitude(image)
    print('gx_gradient: ' + str(gx_gradient[28][28]))

    gy_gradient = compute_vertical_gradient_magnitude(image)
    print('gy_gradient: ' + str(gy_gradient[28][28]))

    magnitude = compute_gradient_magnitude(gx_gradient, gy_gradient)
    print('gradient: ' + str(magnitude[28][28]))

    theta = compute_gradient_angle(magnitude, gx_gradient, gy_gradient)
    print('angle: ' + str(theta[28][28]))

    feature_vector = compute_hog_feature(theta, magnitude)
    items = len(feature_vector) * len(feature_vector[0])
    print('total items: ' + str(items))


def compute_hog_feature(theta, magnitude):
    # sanity check
    assert len(theta) == len(magnitude)
    assert len(theta[0]) == len(magnitude[0])

    print('dimensions of theta: ' + str(len(theta)) + " x " + str(len(theta[0])))

    # final result
    hog_feature = []

    cell_size = 8
    block_size = 16

    cells = 0
    blocks = 0

    for i in range(0, len(theta) - cell_size, cell_size):
        for j in range(0, len(theta[i]) - cell_size, cell_size):
            # divide arrays into 16 x 16 subarrays
            theta_block = [[0] * block_size for _ in range(block_size)]
            magnitude_block = [[0] * block_size for _ in range(block_size)]

            for k in range(block_size):
                for l in range(block_size):
                    theta_block[k][l] = theta[i + k][j + l]
                    magnitude_block[k][l] = magnitude[i + k][j + l]

            blocks += 1
            # break each 16 x 16 subarray into 8 x 8 cells
            theta_cell = [[0] * cell_size for _ in range(cell_size)]
            magnitude_cell = [[0] * cell_size for _ in range(cell_size)]

            for m in range(0, len(theta_block), cell_size):
                for n in range(0, len(theta_block), cell_size):
                    # copy into 8 x 8 subarrays
                    for o in range(cell_size):
                        for p in range(cell_size):
                            theta_cell[o][p] = theta_block[m + o][n + p]
                            magnitude_cell[o][p] = magnitude_block[m + o][n + p]

                    # add histogram to hog feature vector
                    cells += 1
                    histogram = Histogram(theta_cell, magnitude_cell)
                    hog_feature.append(histogram.bins)

    print('blocks: ' + str(blocks) + ' | ' + 'cells: ' + str(cells))
    return hog_feature


if __name__ == '__main__':
    detect()
