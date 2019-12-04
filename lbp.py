from lbp_histogram import LbpHistogram


def compute_lbp_feature_histograms(image):
    # start by dividing into non-overlapping 16 x 16 blocks
    lbp_feature = []
    block_size = 16
    blocks = 0

    for i in range(0, len(image), block_size):
        for j in range(0, len(image[i]), block_size):

            image_block = [[0] * block_size for _ in range(block_size)]

            for n in range(len(image_block)):
                for m in range(len(image_block[n])):
                    image_block[n][m] = image[i + n][j + m]

            blocks += 1
            lbp_histogram = LbpHistogram(image_block)
            lbp_feature.append(lbp_histogram.bins)

    print('lbp feature space: ' + str(blocks * 59))
    return lbp_feature
