import math
import random
import copy


# ALERT! number of human / non-human feature vectors must be the same!
class NeuralNetwork:
    def __init__(self, human_feature_vectors, nonhuman_feature_vectors, hidden_layer_neurons):
        self.epochs = 2
        self.learning_rate = 0.2
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
        self.predicted_output = [[0.0] for _ in range(len(self.human_feature_vectors))]
        # FIXME dummy value for output layer delta - needs to be 1 x 10
        self.output_layer_delta = [[0.0] for _ in range(len(self.human_feature_vectors))]
        # FIXME dummy value for hidden layer delta - needs to be a 1 x 10
        self.hidden_layer_delta = [[0.0] for _ in range(len(self.human_feature_vectors))]
        # this will contain sum of all errors
        self.average_error = 0.0
        self.train()

    def train(self):
        # train positive first
        for i in range(self.epochs):
            print('human iteration: ' + str(i))
            self.feed_forward(self.human_feature_vectors)
            # FIXME - needs to be 1 x 10
            target = [[1.0] for _ in range(len(self.human_feature_vectors))]
            self.backpropogate(target)
            self.update(self.human_feature_vectors)

        print('predicted output for human: ' + str(self.predicted_output))

        # TODO reset predicted output?

        for i in range(self.epochs):
            print('non-human iteration: ' + str(i))
            self.feed_forward(self.nonhuman_feature_vectors)
            # FIXME - needs to be 1 x 10
            target = [[0.0] for _ in range(len(self.nonhuman_feature_vectors))]
            self.backpropogate(target)
            self.update(self.nonhuman_feature_vectors)

        print('predicted output for non-human: ' + str(self.predicted_output))

    # sigmoid function for output neuron
    def sigmoid(self, X):
        result = [[0.0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                result[i][j] = 1 / (1 + math.exp(0 - X[i][j]))
        return result

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
        result = [[0.0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] <= 0:
                    result[i][j] = 0.0
                else:
                    result[i][j] = X[i][j]
        return result

    # derivative of ReLU
    def derivative_of_relu(self, X):
        result = [[0.0 for x in range(len(X[0]))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[i])):
                result[i][j] = 1 * (X[i][j] > 0)
        return result

    # forward direction for training
    def feed_forward(self, vectors):
        self.hidden_layer_output = self.relu(self.matrix_add(self.matrix_multiply(vectors, self.hidden_layer_weights), self.hidden_layer_bias))

        self.predicted_output = self.sigmoid(self.matrix_add(self.matrix_multiply(self.hidden_layer_output, self.output_layer_weights),
                                                self.output_layer_bias))

    # backward direction for training
    # target output is an array of 2 x 1 containing 1 for humans and 0 for non-humans
    def backpropogate(self, target_output):
        # check that this output is a 2 x 1 matrix
        # output_layer_errors = target_output - self.predicted_output
        # FIXME - this needs to be 1 x 10 based on number of feature vectors
        output_layer_errors = [[0.0] for _ in range(len(self.human_feature_vectors))]
        for i in range(len(target_output)):
            output_layer_errors[i][0] = target_output[i][0] - self.predicted_output[i][0]

        # TODO - compute average error here! take absolute value

        sigmoid = self.derivative_of_sigmoid(self.predicted_output)

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

    def predict(self, vector):
        print('dimensions of vector: ' + str(len(vector)) + ' x ' + str(len(vector[0])))
        print('dimensions of hidden layer weights: ' + str(len(self.hidden_layer_weights)) + ' x ' + str(len(self.hidden_layer_weights[0])))
        print('dimensions of hidden layer bias: ' + str(len(self.hidden_layer_bias)) + ' x ' + str(
            len(self.hidden_layer_bias[0])))
        hidden_layer_output = self.relu(self.matrix_add(self.matrix_multiply(vector, self.hidden_layer_weights), self.hidden_layer_bias))
        predicted_output = self.sigmoid(self.matrix_add(self.matrix_multiply(hidden_layer_output, self.output_layer_weights),
                            self.output_layer_bias))
        return predicted_output
