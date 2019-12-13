import math
import random


class HogNeuralNetwork:
    def __init__(self, human_feature_vectors, nonhuman_feature_vectors, hidden_layer_neurons):
        self.epochs = 10
        self.learning_rate = 0.1
        self.human_feature_vectors = human_feature_vectors
        print('human feature vectors dimensions: ' + str(len(self.human_feature_vectors)) + ' x ' + str(len(self.human_feature_vectors[0])))
        self.nonhuman_feature_vectors = nonhuman_feature_vectors
        # hidden_layer_neurons should be 200 or 400
        self.hidden_layer_neurons = hidden_layer_neurons
        # hidden layer weights should be 7524 x 200
        self.hidden_layer_weights = self.create_random_matrix(self.hidden_layer_neurons, len(self.human_feature_vectors[0]))
        # hidden layer bias should be 1 x 200
        self.hidden_layer_bias = [[-1.0] * self.hidden_layer_neurons] * 1
        # output layer weights should be 200 x 1
        self.output_layer_weights = self.create_random_matrix(1, self.hidden_layer_neurons)

        self.output_layer_bias = [[-1.0]]
        # dummy value for hidden_layer_output
        self.hidden_layer_output = [0.0]
        # dummy value for predicted output
        self.predicted_output = [[0.0]]
        # FIXME dummy value for output layer delta - needs to be 1 x 10
        self.output_layer_delta = [[0.0], [0.0]]
        # FIXME dummy value for hidden layer delta - needs to be a 1 x 10
        self.hidden_layer_delta = [[0.0], [0.0]]
        # this will contain sum of all errors
        self.average_error = 0.0

        self.train()

    def train(self):
        # train positive first
        for i in range(self.epochs):
            print('iteration: ' + str(i))
            self.feed_forward(self.human_feature_vectors)
            # FIXME - needs to be 1 x 10
            target = [[1], [1]]
            self.backpropogate(target)
            self.update(self.human_feature_vectors)

        print('predicted output for human: ' + str(self.predicted_output))

        # TODO - anything we need to save for human should be saved here!

        # reset values
        self.hidden_layer_weights = self.create_random_matrix(self.hidden_layer_neurons, len(self.human_feature_vectors[0]))
        self.output_layer_weights = self.create_random_matrix(1, self.hidden_layer_neurons)
        self.hidden_layer_bias = [[-1.0] * self.hidden_layer_neurons] * 1
        self.output_layer_bias = [[-1.0]]
        self.predicted_output = [[0.0]]

        for i in range(self.epochs):
            print('iteration: ' + str(i))
            self.feed_forward(self.nonhuman_feature_vectors)
            # FIXME - needs to be 1 x 10
            target = [[0], [0]]
            self.backpropogate(target)
            self.update(self.nonhuman_feature_vectors)

        print('predicted output for non-human: ' + str(self.predicted_output))

        # TODO - anything we need to save for non-human should be saved here!

    # sigmoid function for output neuron
    def sigmoid(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = 1 / (1 + math.exp(0 - X[i][j]))
        return X

    # derivative of sigmoid
    def derivative_of_sigmoid(self, X):
        # this needs to return a copy
        result = [[0.0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                result[i][j] = X[i][j] * (1 - X[i][j])
        return result

    # ReLu function for hidden neurons
    def relu(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] <= 0:
                    X[i][j] = 0.0
        return X

    # derivative of ReLU
    def derivative_of_relu(self, X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = 1 * (X[i][j] > 0)
        return X

    # forward direction for training
    def feed_forward(self, vectors):
        self.hidden_layer_output = self.relu(self.matrix_add(self.matrix_multiply(vectors, self.hidden_layer_weights), self.hidden_layer_bias))

        self.predicted_output = self.sigmoid(self.matrix_add(self.matrix_multiply(self.hidden_layer_output, self.output_layer_weights),
                                                  self.output_layer_bias))

        print('sigmoid predicted output: ' + str(self.predicted_output))

    # backward direction for training
    # target output is an array of 2 x 1 containing 1 for humans and 0 for non-humans
    def backpropogate(self, target_output):
        # check that this output is a 2 x 1 matrix
        # output_layer_errors = target_output - self.predicted_output
        # FIXME - this needs to be 1 x 10 based on number of feature vectors
        output_layer_errors = [[0.0], [0.0]]
        for i in range(len(target_output)):
            output_layer_errors[i][0] = target_output[i][0] - self.predicted_output[i][0]

        sigmoid = self.derivative_of_sigmoid(self.predicted_output)

        print('derivative of sigmoid predicted output: ' + str(sigmoid))
        for i in range(len(sigmoid)):
            for j in range(len(sigmoid[i])):
                self.output_layer_delta[i][j] = output_layer_errors[i][j] * sigmoid[i][j]

        hidden_layer_errors = self.matrix_multiply(self.output_layer_delta, self.transpose(self.output_layer_weights))

        self.hidden_layer_delta = self.multiply(hidden_layer_errors, self.derivative_of_relu(self.hidden_layer_output))

    def update(self, inputs):
        # update hidden layer weights and hidden layer / output layer bias
        # output layer first
        transposed = self.transpose(self.hidden_layer_output)
        test_matrix = self.matrix_multiply(transposed, self.output_layer_delta)

        for i in range(len(test_matrix)):
            for j in range(len(test_matrix[i])):
                test_matrix[i][j] = test_matrix[i][j] * self.learning_rate

        # assert that test_matrix is 1 x 200
        self.output_layer_weights = self.matrix_add(self.output_layer_weights, test_matrix)

        sum = 0
        for k in range(len(self.output_layer_delta)):
            for l in range(len(self.output_layer_delta[k])):
                sum += self.output_layer_delta[k][l]
        result = [[sum * self.learning_rate]]

        self.output_layer_bias = self.matrix_add(self.output_layer_bias, result)

        # now update hidden layer
        test_matrix = self.matrix_multiply(self.transpose(inputs), self.hidden_layer_delta)
        for i in range(len(test_matrix)):
            for j in range(len(test_matrix[i])):
                test_matrix[i][j] = test_matrix[i][j] * self.learning_rate

        # assert that test_matrix is 1 x 200
        self.hidden_layer_weights = self.matrix_add(self.hidden_layer_weights, test_matrix)

        sum = 0
        for k in range(len(self.hidden_layer_delta)):
            for l in range(len(self.hidden_layer_delta[k])):
                sum += self.hidden_layer_delta[k][l]
        result = sum * self.learning_rate

        # self.hidden_layer_bias = self.matrix_add(self.hidden_layer_bias, result)
        for i in range(len(self.hidden_layer_bias)):
            for j in range(len(self.hidden_layer_bias[i])):
                self.hidden_layer_bias[i][j] += result

    def multiply(self, X, Y):
        # this is not a matrix multiply ... arrays in this method should have same dimensions
        result = [[0.0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                result[i][j] = X[i][j] * Y[i][j]
        return result

    def matrix_multiply(self, X, Y):
        # print('dimensions of X: ' + str(len(X)) + ' x ' + str(len(X[0])))
        # print('dimensions of Y: ' + str(len(Y)) + ' x ' + str(len(Y[0])))
        result = [[0.0 for x in range(len(Y[0]))] for y in range(len(X))]
        for i in range(len(X)):
            # iterate through columns of Y
            for j in range(len(Y[0])):
                # iterate through rows of Y
                for k in range(len(Y)):
                    result[i][j] = X[i][k] * Y[k][j]
        return result

    def matrix_add(self, X, Y):
        # add Y to X ... X is 2d, Y is 1d
        # print('dimensions of X: ' + str(len(X)) + ' x ' + str(len(X[0])))
        # print('dimensions of Y: ' + str(len(Y)) + ' x ' + str(len(Y[0])))
        result = [[0.0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                result[i][j] = X[i][j] + Y[0][j]
        return result

    def create_random_matrix(self, length, width):
        random_matrix = [[0.0] * length for _ in range(width)]
        for i in range(len(random_matrix)):
            for j in range(len(random_matrix[i])):
                random_matrix[i][j] = random.uniform(-0.5, 0.5)
        return random_matrix

    # check if correct
    def transpose(self, X):
        result = []
        for row in range(len(X)):
            result = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
        return result
