import numpy as np
import random


### Initialize the model's parameters

def initialize_parameters(L, n_h, seed, sigma):
    """
    Argument:
    L   -- size of the system
    n_h -- size of the hidden layer
    seed -- random seed
    sigma -- sigma parameter of Gaussian distribution
    
    Returns:
    parameters -- python dictionary containing the complex parameters:
                    W1 -- weight matrix of the visible (0-th) layer, shape (n_h, L)
                    b1 -- bias matrix of the visible (0-th) layer, shape (n_h, 1)
    """
    np.random.seed(seed)

    parameters = {}
    parameters['W1'] = np.zeros((n_h, L), dtype=complex)
    for i in range(n_h):
        for j in range(L):
            parameters['W1'].itemset((i,j), random.gauss(0,sigma) + random.gauss(0,sigma) * 1j)

    parameters['b1'] = np.zeros((n_h, 1), dtype=complex)

    
    assert(parameters['W1'].shape == (n_h, L))
    assert(parameters['b1'].shape == (n_h, 1))
        
    return parameters




### Forward Propagation

# ReLU

def relu(Z):
    """
    Implement the ReLU function.
    
    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    Z -- stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0 , Z)
    

    assert(A.shape == Z.shape)
    

    return A, Z

# CReLU

def crelu(Z):
    """ 
    Implement the ReLU function. Reference: deep complex net.

    Arguments:
    Z -- Complex Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    Z -- stored for computing the backward pass efficiently
    """

    A = np.copy(Z)
    length = len(Z)
    width = len(Z[0])

    for i in range(length):
        for j in range(width):
            A.itemset((i,j), max(0, Z[i,j].real + max(0, Z[i,j].imag * 1j)))

    assert(A.shape == Z.shape)

    return A, Z

# Leaky Relu

def leaky_relu(Z):
    """
    Implement the leaky RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0.01*Z,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

# Sigmoid

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    Z -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    
    
    return A, Z


# Tanh

def tanh(Z):
    """
    Implements the tanh activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of tranh(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = np.tanh(Z)
    

    assert(A.shape == Z.shape)
    
    return A, Z

# Log(Cosh) modified according to the Carleo netket

def real_lncosh_modified(Z):
    """
    Implement the MODIFIED lncosh activation real np matrix, in order to avoid the "inf problem".
    Reference: Carleo netket

    Arguments:
    Z -- REAL numpy array of any shape
    
    Returns:
    real_lncosh_modified(Z) -- np matrix with the same shape as Z
    """

    # return np.where( np.abs(Z) <= 12. , real_lncosh(Z), np.abs(Z) - np.log(2.) )
    return np.where( np.abs(Z) <= 12. , np.log(np.cosh(Z)), np.abs(Z) - np.log(2) )


def complex_lncosh(Z):
    """
    Implements the COMPLEX ln(cosh) activation in numpy.
    Reference: Carleo netket
    
    Arguments:
    Z -- COMPLEX numpy array of any shape
    
    Returns:
    A -- output of lncosh(z), same shape as Z
    Z -- returns Z as well, useful during backpropagation
    """
    
    A = real_lncosh_modified(np.real(Z)) + np.log( np.cos(np.imag(Z)) + 1j * np.tanh(np.real(Z)) * np.sin(np.imag(Z)))
    
    return A, Z


# Feedforward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing Z;
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = np.dot(W, A_prev) + b
        A, cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = np.dot(W, A_prev) + b
        A, cache = relu(Z)

    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = np.dot(W, A_prev) + b
        A, cache = tanh(Z)

    elif activation == "leaky_relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = np.dot(W, A_prev) + b
        A, cache = leaky_relu(Z)

    elif activation == "lncosh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = np.dot(W, A_prev) + b
        A, cache = complex_lncosh(Z)

    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    return A, cache



### Feedforward Result of 2_layer FFNN model

def FFNN_paper_model(X, parameters):
    """
    Implement the forward propagation of the 2-layer FFNN model

    Arguments:
    X -- |sigma> (input data)
    parameters -- python dictionary containing your parameters: W1, b1
    
    Returns:
    phi -- phi(|sigma>, parameters) = e^Y (approximated wavefunction of the input spin configuration)
    Y -- Result of deepnet model
    Z -- W1 * X + b1, stored for computing the backward pass efficiently
    """
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    
    # Feedforward
    A, Z = linear_activation_forward(X, W1, b1, 'lncosh')
    Y = np.sum(A, keepdims=True)
    phi = np.exp(Y)
    phi = np.squeeze(phi)

    return phi, Y, Z


### Test

# L = 6

# # input
# X = np.array([-1, 1, 1, -1, -1, 1]).reshape((L,1))

# # Initialize parameters, then retrieve W1, b1, W2, b2. 
# parameters = initialize_parameters(L, n_h = 2*L, seed=1234, sigma=0.01)
# # print(parameters)


# # Feedforward
# A, Z = linear_activation_forward(X, parameters['W1'], parameters['b1'], 'logcosh')
# print(Z)
# print(A)
# Y= np.sum(A, keepdims=True)
# print(Y)
# # print(phi)


