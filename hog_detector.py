from image_converter import get_image_array, convert_grayscale
from sobel_operator import compute_gradient_magnitude, compute_horizontal_gradient_magnitude, \
    compute_vertical_gradient_magnitude, compute_gradient_angle
from histogram import Histogram
from lbp import compute_lbp_feature_histograms
from neural_network import HogNeuralNetwork


def detect():
    # positive image
    image = get_image_array('C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive\\crop_000010b.bmp')
    gx_gradient = compute_horizontal_gradient_magnitude(image)
    print('gx_gradient: ' + str(gx_gradient[28][28]))

    gy_gradient = compute_vertical_gradient_magnitude(image)
    print('gy_gradient: ' + str(gy_gradient[28][28]))

    magnitude = compute_gradient_magnitude(gx_gradient, gy_gradient)
    print('gradient: ' + str(magnitude[28][28]))

    theta = compute_gradient_angle(magnitude, gx_gradient, gy_gradient)
    print('angle: ' + str(theta[28][28]))

    hog_feature_vector_1 = compute_hog_feature(theta, magnitude)

    # second positive image
    image = get_image_array(
        'C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive\\crop001030c.bmp')
    gx_gradient = compute_horizontal_gradient_magnitude(image)
    print('gx_gradient: ' + str(gx_gradient[28][28]))

    gy_gradient = compute_vertical_gradient_magnitude(image)
    print('gy_gradient: ' + str(gy_gradient[28][28]))

    magnitude = compute_gradient_magnitude(gx_gradient, gy_gradient)
    print('gradient: ' + str(magnitude[28][28]))

    theta = compute_gradient_angle(magnitude, gx_gradient, gy_gradient)
    print('angle: ' + str(theta[28][28]))

    hog_feature_vector_2 = compute_hog_feature(theta, magnitude)

    # negative image
    neg_image = get_image_array('C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_negative\\01-03e_cut.bmp')
    neg_gx_gradient = compute_horizontal_gradient_magnitude(neg_image)
    # print('gx_gradient: ' + str(gx_gradient[28][28]))

    neg_gy_gradient = compute_vertical_gradient_magnitude(neg_image)
    # print('gy_gradient: ' + str(gy_gradient[28][28]))

    neg_magnitude = compute_gradient_magnitude(neg_gx_gradient, neg_gy_gradient)
    # print('gradient: ' + str(magnitude[28][28]))

    neg_theta = compute_gradient_angle(neg_magnitude, neg_gx_gradient, neg_gy_gradient)
    # print('angle: ' + str(theta[28][28]))

    neg_hog_feature_vector = compute_hog_feature(neg_theta, neg_magnitude)


    # hog_items = len(hog_feature_vector) * len(hog_feature_vector[0]) * len(hog_feature_vector[0][0])
    # print('hog feature dimensions: ' + str(len(hog_feature_vector)) + ' x ' + str(len(hog_feature_vector[0])) + ' x '
    #       + str(len(hog_feature_vector[0][0])))
    # print('hog feature space: ' + str(hog_items))
    # print(str(hog_feature_vector))

    # assert all values are between 0 and 1
    # for i in range(len(hog_feature_vector)):
    #     for j in range(len(hog_feature_vector[i])):
    #         for k in range(len(hog_feature_vector[i][j])):
    #             assert hog_feature_vector[i][j][k] >= 0 and hog_feature_vector[i][j][k] <= 1

    lbp_feature_vector = compute_lbp_feature_histograms(image)
    # print(str(lbp_feature_vector))

    # assert all values are between 0 and 1
    # for i in range(len(lbp_feature_vector)):
    #     for j in range(len(lbp_feature_vector[i])):
    #         assert lbp_feature_vector[i][j] >= 0 and lbp_feature_vector[i][j] <= 1

    # the hog feature vector is 3D ... the lbp feature vector is 2D
    # TODO - write a function which generates an hog feature vector for all training images (10 positive / 10 negative)
    positive_hog_feature_vectors = [hog_feature_vector_1, hog_feature_vector_2]
    negative_hog_feature_vectors = [neg_hog_feature_vector, neg_hog_feature_vector]

    network = HogNeuralNetwork(positive_hog_feature_vectors, negative_hog_feature_vectors, 200)


def compute_hog_feature(theta, magnitude):
    # sanity check
    assert len(theta) == len(magnitude)
    assert len(theta[0]) == len(magnitude[0])

    print('dimensions of theta: ' + str(len(theta)) + " x " + str(len(theta[0])))

    # final result
    hog_feature = []

    cell_size = 8
    block_size = 16

    for i in range(0, len(theta) - cell_size, cell_size):
        for j in range(0, len(theta[i]) - cell_size, cell_size):
            # divide arrays into 16 x 16 subarrays
            theta_block = [[0] * block_size for _ in range(block_size)]
            magnitude_block = [[0] * block_size for _ in range(block_size)]

            for k in range(block_size):
                for l in range(block_size):
                    theta_block[k][l] = theta[i + k][j + l]
                    magnitude_block[k][l] = magnitude[i + k][j + l]

            histogram = Histogram(theta_block, magnitude_block)

            for item in histogram.flattened:
                hog_feature.append(item)

    return hog_feature


if __name__ == '__main__':
    detect()
