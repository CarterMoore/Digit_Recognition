import numpy as np

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Creates numpy array of biases for each layer after input layer
        # Gets random value from normal distribution
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]] 
        # Creates list of numpy matrices with weights connecting each layer
        # weights[0] will be weights connecting first and second layers
        # each matrix has number of neurons in second layer for rows and first layer for columns
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, activation):
        '''Takes activations from input layer and uses weights and biases
        to compute activation for each subsequent layer'''
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid(w@activation+b)
        return activation

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases)]
        nabla_w = [np.zeros(w.shape) for w in self.weights)]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nable_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

def sigmoid(z, derivative=False):
    '''Applies sigmoid function elementwise'''
    if derivative:
        return np.exp(-z) * (sigmoid(z)**2)
    return 1/(1 + np.exp(-z))

if __name__ == '__main__':
    n = Network([2, 3, 5])
    input_a = [0.8, 0.42]
    output = n.feedforward(input_a)
    print(output)

