import numpy


class Histogram:
    def __init__(self, theta_block, magnitude_block):
        self.magnitude_block = magnitude_block
        self.theta_block = theta_block
        self.centers = [0, 20, 40, 60, 80, 100, 120, 140, 160]
        # 4 x 9 bins matrix: each 16 x 16 block becomes four 8 x 8 cells
        self.bins = [[0.0] * 9 for _ in range(4)]
        self.cell_size = 8
        # four 8 x 8 cells matrices
        self.theta_cells = []
        self.magnitude_cells = []
        self.convert_blocks_to_cells()
        self.compute_feature_vector()
        self.normalize()

    def normalize(self):
        # TODO - flatten bins and apply L2 norm
        return

    def convert_blocks_to_cells(self):
        for m in range(0, len(self.theta_block), self.cell_size):
            for n in range(0, len(self.theta_block), self.cell_size):
                # copy into 8 x 8 subarrays
                theta_cell = [[0] * self.cell_size for _ in range(self.cell_size)]
                magnitude_cell = [[0] * self.cell_size for _ in range(self.cell_size)]

                for o in range(self.cell_size):
                    for p in range(self.cell_size):
                        theta_cell[o][p] = self.theta_block[m + o][n + p]
                        magnitude_cell[o][p] = self.magnitude_block[m + o][n + p]

                self.theta_cells.append(theta_cell)
                self.magnitude_cells.append(magnitude_cell)

    def compute_feature_vector(self):
        # 4 is the total number of cells
        for i in range(len(self.theta_cells)):
            for j in range(len(self.theta_cells[i])):
                for k in range(len(self.theta_cells[i][j])):
                    self.add_to_bin(self.theta_cells[i][j][k], self.magnitude_cells[i][j][k], self.bins[i])

    # FIXME - just look at bin centers and calculate distance to add % of gradient to two bins
    def add_to_bin(self, angle, magnitude, bin):
        # working with unsigned angles
        if angle >= 180:
            angle -= 180

        # corner case between 1st and 9th bin
        if angle >= 160:
            first_bin = 8
            second_bin = 0
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 0 and angle < 20:
            first_bin = 0
            second_bin = 1
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 20 and angle < 40:
            first_bin = 1
            second_bin = 2
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 40 and angle < 60:
            first_bin = 2
            second_bin = 3
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 60 and angle < 80:
            first_bin = 3
            second_bin = 4
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 80 and angle < 100:
            first_bin = 4
            second_bin = 5
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 100 and angle < 120:
            first_bin = 5
            second_bin = 6
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 120 and angle < 140:
            first_bin = 6
            second_bin = 7
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

        if angle >= 140 and angle < 160:
            first_bin = 7
            second_bin = 8
            percentage = self.calculate_distance(angle, self.centers[first_bin])
            bin[first_bin] += (100 - percentage) * magnitude
            bin[second_bin] += percentage * magnitude

    def calculate_distance(self, angle, center):
        # calculate what percentage of the angle belongs to a given center (20 is static distance between centers)
        percentage = numpy.absolute(angle - center) / 20
        return percentage
