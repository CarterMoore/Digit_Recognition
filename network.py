import numpy as np
import random

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
            activation = sigmoid(np.dot(w,activation)+b)
        return activation

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''Train network with stochastic gradient descent, training_data is list of tuples (x,y) with 
        training inputs and required outputs.'''
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
        '''Perform gradient descent on one mini batch. mini_batch is list of tuples, eta is learning rate'''
        nabla_b = [np.zeros(b.shape) for b in self.biases] # Gradient for biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Gradient for weights
        # Compute gradient for each x,y pair and add to total gradient
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Average gradient and multiply by learning rate, then use to update weights and biases
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid(z_vectors[-1], True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,np.transpose(activations[-2]))
        for l in range(2, self.num_layers):
            z = z_vectors[-l]
            sp = sigmoid(z, True)
            delta = np.dot(np.transpose(self.weights[-l+1]),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,np.transpose(activations[-l-1]))
        return(nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z, derivative=False):
    '''Applies sigmoid function elementwise'''
    if derivative:
        return np.exp(-z) * (sigmoid(z)**2)
    return 1/(1 + np.exp(-z))

if __name__ == '__main__':
    net = Network([1,2,1])
    inputs = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]
    print(np.array(inputs[0]))
