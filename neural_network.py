from itertools import chain
import math
import random


class HogNeuralNetwork:
    def __init__(self, human_feature_vectors, nonhuman_feature_vectors, hidden_layer_neurons):
        self.epochs = 1000
        self.learning_rate = 0.1
        self.human_feature_vectors = self.flatten(human_feature_vectors)
        print('human feature vectors: ' + str(self.human_feature_vectors))
        self.nonhuman_feature_vectors = self.flatten(nonhuman_feature_vectors)
        # hidden_layer_neurons should be 200 or 400
        self.hidden_layer_neurons = hidden_layer_neurons
        # hidden layer weights should be 7524 x 200
        self.hidden_layer_weights = self.create_random_matrix(self.hidden_layer_neurons, len(self.human_feature_vectors[0]))
        print('dimensions of hidden layer weights: ' + str(len(self.hidden_layer_weights)) + ' x '
              + str(len(self.hidden_layer_weights[0])))
        # hidden layer bias should be 1 x 200
        self.hidden_layer_bias = [[-1.0] * self.hidden_layer_neurons] * 1
        print('dimensions of hidden layer bias: ' + str(len(self.hidden_layer_bias)) + ' x ' + str(len(self.hidden_layer_bias[0])))
        # output layer weights should be 200 x 1
        # self.output_layer_weights = [[random.uniform(0, 1)] for _ in range(self.hidden_layer_neurons)]
        self.output_layer_weights = self.create_random_matrix(1, self.hidden_layer_neurons)
        print('dimensions of output layer weights: ' + str(len(self.output_layer_weights)) + ' x ' + str(len(self.output_layer_weights[0])))
        self.output_layer_bias = [[-1]]
        # dummy value for hidden_layer_output
        self.hidden_layer_output = [0]
        # dummy value for predicted output
        self.predicted_output = [0]
        # dummy value for output layer delta
        self.output_layer_delta = [0]
        # dummy value for hidden layer delta
        self.hidden_layer_delta = [0]
        # this will contain sum of all errors
        self.error = 0
        self.feed_forward()

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

    # sigmoid function for output neuron
    def sigmoid(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = 1 / (1 + math.exp(0 - X[i][j]))
        return X

    # derivative of sigmoid
    # FIXME - x is now a list!
    def derivative_of_sigmoid(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = X[i][j] * (1 - X[i][j])
        return X

    # ReLu function for hidden neurons
    def relu(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] <= 0:
                    X[i][j] = 0
        return X

    # derivative of ReLU
    # FIXME - x is now a list!
    def derivative_of_relu(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = 1 * (X[i][j] > 0)
        return X

    # forward direction for training
    def feed_forward(self):
        hidden_layer_activation = self.matrix_add(self.matrix_multiply(self.human_feature_vectors, self.hidden_layer_weights), self.hidden_layer_bias)
        # hidden layer activation should be 2 x 200
        print('hidden layer activation: ' + str(hidden_layer_activation))
        print('hidden layer activation dimensions: ' + str(len(hidden_layer_activation)) + ' x '
              + str(len(hidden_layer_activation[0])))
        self.hidden_layer_output = self.relu(hidden_layer_activation)
        print('hidden layer output: ' + str(self.hidden_layer_output))
        output_layer_activation = self.matrix_add(self.matrix_multiply(self.hidden_layer_output, self.output_layer_weights),
                                                  self.output_layer_bias)
        self.predicted_output = self.sigmoid(output_layer_activation)
        print('predicted output: ' + str(self.predicted_output))
        print('predicted output dimensions: ' + str(len(self.predicted_output)) + ' x ' + str(len(self.predicted_output[0])))

    # backward direction for training
    # target output is an array of 2 x 1 containing 1 for humans and 0 for non-humans
    def backpropogate(self, target_output):
        # check that this output is a 2 x 1 matrix
        # this wont work
        output_layer_errors = target_output - self.predicted_output
        self.output_layer_delta = self.matrix_multiply(output_layer_errors,
                                                       self.derivative_of_sigmoid(self.predicted_output))
        hidden_layer_errors = self.matrix_multiply(self.output_layer_delta, self.transpose(self.output_layer_weights))
        self.hidden_layer_delta = self.matrix_multiply(hidden_layer_errors,
                                                       self.derivative_of_relu(self.hidden_layer_output))

    def update(self):
        return

    def matrix_multiply(self, X, Y):
        # print('dimensions of X: ' + str(len(X)) + ' x ' + str(len(X[0])))
        # print('dimensions of Y: ' + str(len(Y)) + ' x ' + str(len(Y[0])))
        result = [[0 for x in range(len(Y[0]))] for y in range(len(X))]
        for i in range(len(X)):
            # iterate through columns of Y
            for j in range(len(Y[0])):
                # iterate through rows of Y
                for k in range(len(Y)):
                    result[i][j] += X[i][k] * Y[k][j]
        return result

    def matrix_add(self, X, Y):
        # add Y to X ... X is 2d, Y is 1d
        # print('dimensions of X: ' + str(len(X)) + ' x ' + str(len(X[0])))
        # print('dimensions of Y: ' + str(len(Y)) + ' x ' + str(len(Y[0])))
        result = [[0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                result[i][j] = X[i][j] + Y[0][j]
        return result

    def create_random_matrix(self, length, width):
        random_matrix = [[0.0] * length for _ in range(width)]
        for i in range(len(random_matrix)):
            for j in range(len(random_matrix[i])):
                random_matrix[i][j] = random.uniform(0, 1)
        return random_matrix

    # check if correct
    def transpose(self, X):
        return list(map(list, zip(*X)))