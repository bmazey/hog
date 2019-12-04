from lbp_histogram import LbpHistogram


def compute_lbp_feature_histograms(image):
    # start by dividing into non-overlapping 16 x 16 blocks
    lbp_feature = []
    block_size = 16

    for i in range(0, len(image), block_size):
        for j in range(0, len(image[i]), block_size):

            image_block = [[0] * block_size for _ in range(block_size)]

            for n in range(len(image_block)):
                for m in range(len(image_block[n])):
                    image_block[n][m] = image[i + n][j + m]

            lbp_histogram = LbpHistogram(image_block)
            lbp_feature.append(lbp_histogram.bins)

    return lbp_feature
