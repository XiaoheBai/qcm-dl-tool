import numpy as np
import deepnet
import cost_function
import matplotlib.pyplot as plt


### Derivative of activation function

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def lncosh_backward(dA, Z):
    """
    Implement the backward propagation for a single LNCOSH unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    dZ = dA * np.tanh(Z)
    
    assert (dZ.shape == Z.shape)
    
    return dZ






### Gradient of Parameters

# Backpropagation

def linear_activation_backward(dA, Z_cache, A_prev, W, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    Note: dW = partial Y / partial W
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- 'Z' we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of Y with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of Y with respect to W (current layer l), same shape as W
    db -- Gradient of Y with respect to b (current layer l), same shape as b
    """
    
    if activation == "relu":
        
        dZ = relu_backward(dA, Z_cache)
        
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, Z_cache)

    elif activation == "lncosh":

        dZ = lncosh_backward(dA, Z_cache)

    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dZ, dW, db, dA_prev


# Define the O Operator

def compute_O_operator(X, Z, parameters):
    """
    Compute the O Operator for each spin configuration X

    Arguments:
    X -- |sigma>
    Z -- Z(|sigma>, parameters) = W1 * X + b1
    parameters -- W1, b1

    Returns:
    O_W1 -- O Operator applied on W1 (matrix of W1.shape)
    O_b1 -- O Operator applied on b1 (matrix of b1.shape)
    """
    L = X.shape[0]
    
    W1 = parameters['W1']
    b1 = parameters['b1']

    dA = np.ones((2*L, 1))
    dZ = lncosh_backward(dA, Z)
    dW1 = np.dot(dZ, X.T)
    db1 = np.sum(dZ, axis = 1, keepdims = True)

    O_W1 = dW1
    O_b1 = db1

    assert (O_W1.shape == W1.shape)
    assert (O_b1.shape == b1.shape)

    return O_W1, O_b1


# Compute the Gradient of the cost (Hamiltonian) with respect to Parameters

def compute_f(train_set, train_set_h_local, parameters, hamiltonian):
    """
    Arguments:
    train_set -- Training set with each row representing one sigma
    train_set_h_local -- h_local list tracked for each |sigma>, stored for computing the gradient descent efficiently
    parameters -- python dictionary containing your parameters W1, b1
    hamiltonian -- this equals to quant_ave_EL <EL>

    Returns:
    f -- python dictionary containing gradient of H with respect to parameters
            f_W1 -- Gradient of H with respect to W1
            f_b1 -- Gradient of H with respect to b1
    """

    n_sample = len(train_set)

    quant_ave_OW1_EL = np.zeros((parameters['W1'].shape[0], L))
    quant_ave_Ob1_EL = np.zeros((parameters['b1'].shape[0], 1))

    quant_ave_OW1 = np.zeros((parameters['W1'].shape[0], L))
    quant_ave_Ob1 = np.zeros((parameters['b1'].shape[0], 1))
    

    for i in range(n_sample):

        spin_configuration = train_set[i].reshape((L,1))
        _, Z = deepnet.FFNN_paper_model(spin_configuration, parameters)
        O_W1, O_b1 = compute_O_operator(spin_configuration, Z, parameters)
        
        
        quant_ave_OW1_EL = quant_ave_OW1_EL + np.conjugate( O_W1 ) * train_set_h_local[i]
        quant_ave_Ob1_EL = quant_ave_Ob1_EL + np.conjugate( O_b1 ) * train_set_h_local[i]

        quant_ave_OW1 = quant_ave_OW1 + np.conjugate( O_W1 )
        quant_ave_Ob1 = quant_ave_Ob1 + np.conjugate( O_b1 )
        
        """ The above average value has not be divided by n_sample !"""
    
    f = {}

    f['f_W1'] = ( quant_ave_OW1_EL - hamiltonian * quant_ave_OW1 )/ n_sample
    f['f_b1'] = ( quant_ave_Ob1_EL - hamiltonian * quant_ave_Ob1 )/ n_sample
    
    
    assert(f['f_W1'].shape == parameters['W1'].shape)
    assert(f['f_b1'].shape == parameters['b1'].shape)

    
    return f






### Update Parameters

# Try the simple gradient descent method

def gd_update_parameters(parameters, f, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    g -- python dictionary containing your gradients, output of compute f
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
   
    parameters['W1'] = parameters['W1'] - learning_rate * f['f_W1']
    parameters['b1'] = parameters['b1'] - learning_rate * f['f_b1']    
    
    return parameters


# Try Adam
# Result: bad. The cost oscillates too heavily.
def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    # L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    # for l in range(L):
    v["dW1"] = np.zeros(parameters['W1'].shape)
    v["db1"] = np.zeros(parameters['b1'].shape)
    s["dW1"] = np.zeros(parameters['W1'].shape)
    s["db1"] = np.zeros(parameters['b1'].shape)
    
    return v, s

def adam_update_parameters(parameters, f, v, s, iter, learning_rate,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    f -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    # L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    # for l in range(L):
    # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
    v["dW1"] = beta1 * v["dW1"] + (1-beta1) * f['f_W1']
    v["db1"] = beta1 * v["db1"] + (1-beta1) * f['f_b1']

    # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
    v_corrected["dW1" ] = v["dW1" ] / (1-beta1**iter)
    v_corrected["db1" ] = v["db1" ] / (1-beta1**iter)

    # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
    s["dW1" ] = beta2 * s["dW1" ] + (1-beta2) * np.power(f['f_W1'], 2)
    s["db1" ] = beta2 * s["db1" ] + (1-beta2) * np.power(f['f_b1'], 2)

    # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
    s_corrected["dW1" ] = s["dW1" ] / (1-beta2**iter)
    s_corrected["db1" ] = s["db1" ] / (1-beta2**iter)

    # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
    parameters["W1" ] = parameters["W1" ] - learning_rate * v_corrected["dW1" ] / (np.power(s_corrected["dW1" ], 1/2) + epsilon)
    parameters["b1" ] = parameters["b1" ] - learning_rate * v_corrected["db1" ] / (np.power(s_corrected["db1" ], 1/2) + epsilon)
    

    return parameters, v, s






### Test
if __name__ == '__main__':

    optimization_method = 'gd'

    L = 6
    # input
    X = np.array([-1, -1, 1, 1, -1, 1]).reshape((L,1))

    # Initialize parameters, then retrieve W1, b1, W2, b2. 
    parameters = deepnet.initialize_parameters(L, n_h = 2*L, seed=1234, sigma=0.01)
    
    if optimization_method == 'adam':
        v, s = initialize_adam(parameters)
    # Feedforward
    # Y, Z = deepnet.FFNN_paper_model(X, parameters)

    # Compute O operator
    # O_W1, O_b1 = compute_o_operator(X, Z, parameters)
    # print('O_W1:', O_W1)
    # print('O_b1:', O_b1)

    
    costs = []
    num_iterations = 200

    # Try the simple Gradient descent optimization method
    for iter in range(1, num_iterations):
        # Feedforward

        Y, Z = deepnet.FFNN_paper_model(X, parameters)

        # Generate train_set

        train_set, train_set_Y = cost_function.markov_chain(X, Y, parameters, n_sample = 1000)

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        """ Finished above and in the process of generating the train_set """
            
        # Compute cost
        hamiltonian, train_set_h_local = cost_function.compute_hamiltonian(train_set, train_set_Y, parameters)

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        f = compute_f(train_set, train_set_h_local, parameters, hamiltonian)
            
        # Update parameters.
        if optimization_method == 'adam':
            parameters, v, s = adam_update_parameters(parameters, f, v, s,
                                                    iter, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8)
        elif optimization_method == 'gd':
            parameters = gd_update_parameters(parameters, f, learning_rate = 0.1)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        # W2 = parameters["W2"]
        # b2 = parameters["b2"]

        # Print the cost
        print("Cost after iteration {}: {}".format(iter, np.real(hamiltonian)))
        costs.append(hamiltonian)
        
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(0.005))
    plt.show()
