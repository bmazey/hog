from itertools import chain


class HogNeuralNetwork:
    def __init__(self, human_feature_vectors, nonhuman_feature_vectors):
        self.epochs = 1000
        self.learning_rate = 0.1
        self.hidden_layer_weights = 0
        self.human_feature_vectors = self.flatten(human_feature_vectors)
        self.nonhuman_feature_vectors = self.flatten(nonhuman_feature_vectors)


    def flatten(self, feature_vector):
        # we have a complicated structure here. we need to flatten 10 3d feature matrices
        # our final dimensions should be 10 x 7524
        flattened = []

        for i in range(len(feature_vector)):
            flat = feature_vector[i]
            # we're going from a 3d matrix to a 1d matrix so we need to flatten twice
            for j in range(2):
                flat = list(chain.from_iterable(flat))

            flattened.append(flat)
        return flattened
