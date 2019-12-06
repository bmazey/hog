from itertools import chain
import numpy


class HogNeuralNetwork:
    def __init__(self, human_feature_vectors, nonhuman_feature_vectors, hidden_layer_neurons):
        self.epochs = 1000
        self.learning_rate = 0.1
        self.human_feature_vectors = self.flatten(human_feature_vectors)
        self.nonhuman_feature_vectors = self.flatten(nonhuman_feature_vectors)
        # hidden_layer_neurons should be 200 or 400
        self.hidden_layer_neurons = hidden_layer_neurons
        # 7524 x 200
        self.hidden_layer_weights = [[numpy.random.randn(0, 1)] * self.hidden_layer_neurons
                                     for _ in range(len(self.human_feature_vectors[0]))]
        # 200 x 1
        self.final_layer_weights = [[numpy.random.randn(0, 1)] for _ in range(self.hidden_layer_neurons)]

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
