

class Histogram:
    def __init__(self, theta, magnitude):
        self.magnitude = magnitude
        self.theta = theta
        self.centers = [0, 20, 40, 60, 80, 100, 120, 140, 160]
        # 4 x 9 bins matrix: each 16 x 16 block becomes four 8 x 8 cells
        self.bins = [[0.0] * 9 for _ in range(4)]
        self.cell_size = 8
        # 8 x 8 x 4 cells matrices
        self.theta_cells = [[[0] * self.cell_size for _ in range(self.cell_size)] for _ in range(4)]
        self.magnitude_cells = [[[0] * self.cell_size for _ in range(self.cell_size)] for _ in range(4)]
        self.compute_feature_vector(theta, magnitude)

    def convert_blocks_to_cells(self, theta, magnitude):


    def compute_feature_vector(self, theta, magnitude):
        for i in range(len(theta)):
            for j in range(len(theta[i])):
                self.add_to_bin(theta[i][j], magnitude[i][j])

    # FIXME - just look at bin centers and calculate distance to add % of gradient to two bins
    def add_to_bin(self, angle, magnitude):
        if angle >= 170 and angle < 350:
            angle -= 180
            # FIXME - handle corner case between bins 1 and 9
            # if angle >= -20 and angle < 0:
                # overlapping bins
        if angle >= -10 and angle < 10:
            self.bins[0] += magnitude

        if angle >= 10 and angle < 30:
            self.bins[1] += magnitude

        if angle >= 30 and angle < 50:
            self.bins[2] += magnitude

        if angle >= 50 and angle < 70:
            self.bins[3] += magnitude

        if angle >= 70 and angle < 90:
            self.bins[4] += magnitude

        if angle >= 90 and angle < 110:
            self.bins[5] += magnitude

        if angle >= 110 and angle < 130:
            self.bins[6] += magnitude

        if angle >= 130 and angle < 150:
            self.bins[7] += magnitude

        if angle >= 150 and angle < 170:
            self.bins[8] += magnitude

        if angle >= 170 or angle < -10:
            print("error in histogram! angle: " + str(angle))
