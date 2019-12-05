

# FIXME - need to normalize by diving by 256 (every value should be between 0 and 1)
class LbpHistogram:
    def __init__(self, image_block):
        self.magnitude = image_block
        # 59 bins: 58 normal transitions, final bin is abnormal
        self.bins = [0.0 for _ in range(59)]
        self.compute_feature_vector(image_block)
        self.normalize()

    def normalize(self):
        for i in range(len(self.bins)):
            self.bins[i] = self.bins[i] / 256

    def compute_feature_vector(self, image_block):
        for i in range(len(image_block)):
            for j in range(len(image_block[i])):
                self.compute_lbp_pattern(i, j, image_block)
        return

    def compute_lbp_pattern(self, i, j, image_block):
        pattern = ''
        try:
            if image_block[i - 1][j - 1] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i - 1][j] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i - 1][j + 1] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i][j + 1] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i + 1][j + 1] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i + 1][j] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i + 1][j - 1] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'
            if image_block[i][j - 1] > image_block[i][j]:
                pattern += '1'
            else:
                pattern += '0'

            self.add_to_bin(pattern)

        except IndexError:
            # default value for border case is 5
            pattern = '00000101'
            self.add_to_bin(pattern)

    def add_to_bin(self, pattern):
        # convert LBP binary pattern to decimal
        decimal = int(pattern, 2)
        patterns = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            6: 5,
            7: 6,
            8: 7,
            12: 8,
            14: 9,
            15: 10,
            16: 11,
            24: 12,
            28: 13,
            30: 14,
            31: 15,
            32: 16,
            48: 17,
            56: 18,
            60: 19,
            62: 20,
            63: 21,
            64: 22,
            96: 23,
            112: 24,
            120: 25,
            124: 26,
            126: 27,
            127: 28,
            128: 29,
            129: 30,
            131: 31,
            135: 32,
            143: 33,
            159: 34,
            191: 35,
            192: 36,
            193: 37,
            195: 38,
            199: 39,
            207: 40,
            223: 41,
            224: 42,
            225: 43,
            227: 44,
            231: 45,
            239: 46,
            240: 47,
            241: 48,
            243: 49,
            247: 50,
            248: 51,
            249: 52,
            251: 53,
            252: 54,
            253: 55,
            254: 56,
            255: 57
        }
        bin_number = patterns.get(decimal, 58)
        # changed from += pixel
        self.bins[bin_number] += 1
