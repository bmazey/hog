import math
import random
import numpy


# ALERT! number of human / non-human feature vectors must be the same!
class NeuralNetwork:
    def __init__(self, human_feature_vectors, nonhuman_feature_vectors, hidden_layer_neurons):
        self.epochs = 1
        self.learning_rate = 0.1
        self.human_feature_vectors = human_feature_vectors
        print('human feature vectors dimensions: ' + str(len(self.human_feature_vectors)) + ' x ' + str(len(self.human_feature_vectors[0])))
        self.nonhuman_feature_vectors = nonhuman_feature_vectors
        # hidden_layer_neurons should be 200 or 400
        self.hidden_layer_neurons = hidden_layer_neurons
        # hidden layer weights should be 7524 x 200
        self.hidden_layer_weights = numpy.array(self.create_random_matrix(self.hidden_layer_neurons, len(self.human_feature_vectors[0])))
        # hidden layer bias should be 1 x 200
        self.hidden_layer_bias = numpy.array([[-1.0] * self.hidden_layer_neurons] * 1)
        # output layer weights should be 200 x 1
        self.output_layer_weights = self.create_random_matrix(1, self.hidden_layer_neurons)

        self.output_layer_bias = [[-1.0]]
        # dummy value for hidden_layer_output
        self.hidden_layer_output = numpy.array([0.0])
        # dummy value for predicted output
        self.predicted_output = numpy.array([0.0])
        # FIXME dummy value for output layer delta - needs to be 1 x 10
        self.output_layer_delta = numpy.array([0.0])
        # FIXME dummy value for hidden layer delta - needs to be a 1 x 10
        self.hidden_layer_delta = numpy.array([0.0])
        # this will contain sum of all errors
        self.average_error = 0.0
        self.train()

    def train(self):
        # train positive first
        for i in range(self.epochs):
            print('iteration: ' + str(i))
            for vector in self.human_feature_vectors:

                self.feed_forward(vector)
                # FIXME - needs to be 1 x 10
                target = [1.0]
                self.backpropogate(target)
                self.update(vector)

            for vector in self.nonhuman_feature_vectors:

                self.feed_forward(vector)
                # FIXME - needs to be 1 x 10
                target = [0.0]
                self.backpropogate(target)
                self.update(vector)

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def derivative_of_sigmoid(self, x):
        return x * (1 - x)

    def relu(self, x):
        y = x.copy()
        y[x < 0] = 0
        return y

    def derivative_of_relu(self, x):
        y = x.copy()
        y[x > 0] = 1
        y[x <= 0] = 0
        return y

    # forward direction for training
    def feed_forward(self, vector):
        input_layer_nodes = numpy.array(vector)
        multiplied = numpy.dot(input_layer_nodes, self.hidden_layer_weights)
        biased = multiplied + self.hidden_layer_bias

        self.hidden_layer_output = self.relu(biased)

        self.predicted_output = self.sigmoid(numpy.dot(self.hidden_layer_output, self.output_layer_weights) + self.output_layer_bias)

    # backward direction for training
    # target output is an array of 2 x 1 containing 1 for humans and 0 for non-humans
    def backpropogate(self, target_output):
        # check that this output is a 2 x 1 matrix
        # output_layer_errors = target_output - self.predicted_output
        # FIXME - this needs to be 1 x 10 based on number of feature vectors
        # output_layer_errors = [[0.0] for _ in range(len(self.human_feature_vectors))]
        #         # for i in range(len(target_output)):
        #         #     output_layer_errors[i][0] = target_output[i][0] - self.predicted_output[i][0]

        output_layer_errors = numpy.array(target_output) - self.predicted_output

        # TODO - compute average error here! take absolute value

        sigmoid = self.derivative_of_sigmoid(self.predicted_output)

        self.output_layer_delta = output_layer_errors * sigmoid

        hidden_layer_errors = numpy.dot(self.output_layer_delta, numpy.transpose(self.output_layer_weights))

        self.hidden_layer_delta = hidden_layer_errors * self.derivative_of_relu(self.hidden_layer_output)

    def update(self, inputs):
        # update hidden layer weights and hidden layer / output layer bias
        # output layer first
        transposed = numpy.transpose(self.hidden_layer_output)
        test_matrix = numpy.dot(transposed, self.output_layer_delta) * self.learning_rate

        # for i in range(len(test_matrix)):
        #     for j in range(len(test_matrix[i])):
        #         test_matrix[i][j] = test_matrix[i][j] * self.learning_rate

        # assert that test_matrix is 1 x 200
        self.output_layer_weights = self.output_layer_weights + test_matrix

        # sum = 0
        # for k in range(len(self.output_layer_delta)):
        #     for l in range(len(self.output_layer_delta[k])):
        #         sum += self.output_layer_delta[k][l]
        # result = [[sum * self.learning_rate]]

        self.output_layer_bias += self.output_layer_delta.sum() * self.learning_rate

        # now update hidden layer
        transposed = numpy.transpose(numpy.array([inputs]))
        # print('dimensions of transposed: ' + str(transposed.shape))
        # print('dimensions of hidden layer delta: ' + str(self.hidden_layer_delta.shape))
        test_matrix = numpy.dot(transposed, self.hidden_layer_delta) * self.learning_rate
        # for i in range(len(test_matrix)):
        #     for j in range(len(test_matrix[i])):
        #         test_matrix[i][j] = test_matrix[i][j] * self.learning_rate

        # assert that test_matrix is 1 x 200
        self.hidden_layer_weights += test_matrix

        # sum = 0
        # for k in range(len(self.hidden_layer_delta)):
        #     for l in range(len(self.hidden_layer_delta[k])):
        #         sum += self.hidden_layer_delta[k][l]
        # result = sum * self.learning_rate
        result = self.hidden_layer_delta.sum() * self.learning_rate

        # self.hidden_layer_bias = self.matrix_add(self.hidden_layer_bias, result)
        # for i in range(len(self.hidden_layer_bias)):
        #     for j in range(len(self.hidden_layer_bias[i])):
        #         self.hidden_layer_bias[i][j] += result
        self.hidden_layer_bias += result

    def create_random_matrix(self, length, width):
        random_matrix = [[0.0] * length for _ in range(width)]
        for i in range(len(random_matrix)):
            for j in range(len(random_matrix[i])):
                random_matrix[i][j] = random.uniform(-0.5, 0.5)
        return random_matrix

    def predict(self, vector):
        input = numpy.array(vector)
        hidden_layer_output = self.relu(numpy.dot(input, self.hidden_layer_weights))
        predicted_output = self.sigmoid(numpy.dot(hidden_layer_output, self.output_layer_weights))
        return predicted_output
