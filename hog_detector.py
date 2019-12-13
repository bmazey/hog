import os
from image_converter import get_image_array, convert_grayscale
from sobel_operator import compute_gradient_magnitude, compute_horizontal_gradient_magnitude, \
    compute_vertical_gradient_magnitude, compute_gradient_angle
from histogram import Histogram
from lbp import compute_lbp_feature_histograms
from neural_network import NeuralNetwork


def detect():

    positive_hog_feature_vectors = generate_hog_feature_vectors(
        'C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive')
    negative_hog_feature_vectors = generate_hog_feature_vectors(
        'C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_negative')

    # this is a test kept here for reference!
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


    # print(str(lbp_feature_vector))

    # assert all values are between 0 and 1
    # for i in range(len(lbp_feature_vector)):
    #     for j in range(len(lbp_feature_vector[i])):
    #         assert lbp_feature_vector[i][j] >= 0 and lbp_feature_vector[i][j] <= 1

    positive_hog_lbp_feature_vectors = generate_hog_lbp_feature_vectors(
        'C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive')
    negative_hog_lbp_feature_vectors = generate_hog_lbp_feature_vectors(
        'C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_negative')

    hog_lbp_network = NeuralNetwork(positive_hog_lbp_feature_vectors, negative_hog_lbp_feature_vectors, 200)

    hog_network = NeuralNetwork(positive_hog_feature_vectors, negative_hog_feature_vectors, 200)

    # TODO - lbp network!
    # lbp_feature_vector = compute_lbp_feature_histograms(image)


def generate_hog_feature_vectors(path):
    # final result
    vectors = []
    files = os.listdir(path)
    for file in files:
        # print(str(file))
        image = get_image_array(path + '\\' + file)
        gx_gradient = compute_horizontal_gradient_magnitude(image)
        gy_gradient = compute_vertical_gradient_magnitude(image)
        magnitude = compute_gradient_magnitude(gx_gradient, gy_gradient)
        theta = compute_gradient_angle(magnitude, gx_gradient, gy_gradient)
        vectors.append(compute_hog_feature(theta, magnitude))
    return vectors


def generate_hog_lbp_feature_vectors(path):
    # final result
    vectors = []
    files = os.listdir(path)
    for file in files:
        # print(str(file))
        image = get_image_array(path + '\\' + file)
        gx_gradient = compute_horizontal_gradient_magnitude(image)
        gy_gradient = compute_vertical_gradient_magnitude(image)
        magnitude = compute_gradient_magnitude(gx_gradient, gy_gradient)
        theta = compute_gradient_angle(magnitude, gx_gradient, gy_gradient)
        hog_vector = compute_hog_feature(theta, magnitude)
        lbp_vector = compute_lbp_feature_histograms(image)
        # print('hog vector: ' + str(hog_vector))
        # print('lbp vector: ' + str(lbp_vector))
        vectors.append(hog_vector + lbp_vector)

    return vectors


def compute_hog_feature(theta, magnitude):
    # sanity check
    assert len(theta) == len(magnitude)
    assert len(theta[0]) == len(magnitude[0])

    # print('dimensions of theta: ' + str(len(theta)) + " x " + str(len(theta[0])))

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
