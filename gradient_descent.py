import numpy as np
import deepnet_FFNN_2
import cost_function


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

    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dZ, dW, db, dA_prev



# Define the O Operator

def compute_o_operator(X, Y, cache1, cache2, A1, parameters):
    """
    Compute the O Operator for each spin configuration X

    Arguments:
    X -- |sigma>
    Y -- phi(|sigma>, parameters)
    parameters -- W1, W2, b1, b2

    Returns:
    O_W1 -- O Operator applied on W1 (matrix of W1.shape)
    O_b1 -- O Operator applied on b1 (matrix of b1.shape)
    O_W2 -- O Operator applied on W2 (matrix of W2.shape)
    O_b2 -- O Operator applied on b2 (matrix of b2.shape)
    """
    
    W2 = parameters['W2']
    W1 = parameters['W1']

    dZ2, dW2, db2, dA1 = linear_activation_backward(1, cache2, A1, W2, 'sigmoid')
    dZ1, dW1, db1, dA0 = linear_activation_backward(dA1, cache1, X, W1, 'relu')

    O_W1 = 1 / Y * dW1
    O_b1 = 1 / Y * db1
    O_W2 = 1 / Y * dW2
    O_b2 = 1 / Y * db1

    return O_W1, O_b1, O_W2, O_b2 


# Compute the Gradient of the cost (Hamiltonian) with respect to Parameters

def compute_gradient(train_set, parameters):
    """
    Arguments:
    train_set -- Training set with each row representing one sigma
    parameters -- python dictionary containing your parameters W1, b1, W2, b2

    Returns:
    delta_W1 -- Gradient of the cost with respect to W1
    delta_b1 -- Gradient of the cost with respect to b1
    delta_W2 -- Gradient of the cost with respect to W2
    delta_b2 -- Gradient of the cost with respect to b2
    """

    n_sample = len(train_set)

    quant_ave_OW1_EL = np.zeros((parameters['W1'].shape[0], L))
    quant_ave_Ob1_EL = np.zeros((parameters['b1'].shape[0], 1))
    quant_ave_OW2_EL = np.zeros((1, parameters['W2'].shape[1]))
    quant_ave_Ob2_EL = np.zeros((1, 1))
    quant_ave_EL = 0
    quant_ave_OW1 = np.zeros((parameters['W1'].shape[0], L))
    quant_ave_Ob1 = np.zeros((parameters['b1'].shape[0], 1))
    quant_ave_OW2 = np.zeros((1, parameters['W2'].shape[1]))
    quant_ave_Ob2 = np.zeros((1, 1))

    for i in range(n_sample):

        spin_configuration = train_set[i].reshape((L,1))
        Y, cache1, cache2, A1 = deepnet_FFNN_2.FFNN_2_model(spin_configuration, parameters)
        O_W1, O_b1, O_W2, O_b2 = compute_o_operator(spin_configuration, Y, cache1, cache2, A1, parameters)
        h_local = cost_function.compute_h_local(spin_configuration, Y, parameters, J1 = 1, J2 = 0.4, epsilon = 0)
        
        quant_ave_OW1_EL = quant_ave_OW1_EL + O_W1 * h_local / n_sample
        quant_ave_Ob1_EL = quant_ave_Ob1_EL + O_b1 * h_local
        quant_ave_OW2_EL = quant_ave_OW2_EL + O_W2 * h_local
        quant_ave_Ob2_EL = quant_ave_Ob2_EL + O_b2 * h_local
        quant_ave_EL += h_local
        quant_ave_OW1 = quant_ave_OW1 + O_W1
        quant_ave_Ob1 = quant_ave_Ob1 + O_b1
        quant_ave_OW2 = quant_ave_OW2 + O_W2
        quant_ave_Ob2 = quant_ave_Ob2 + O_b2
        """ The above average value has not be divided by n_sample !"""

    delta_W1 = 2 / n_sample * ( quant_ave_OW1_EL - quant_ave_EL * quant_ave_OW1 )
    delta_b1 = 2 / n_sample * ( quant_ave_Ob1_EL - quant_ave_EL * quant_ave_Ob1 )
    delta_W2 = 2 / n_sample * ( quant_ave_OW2_EL - quant_ave_EL * quant_ave_OW2 )
    delta_b2 = 2 / n_sample * ( quant_ave_Ob2_EL - quant_ave_EL * quant_ave_Ob2 )
    print(quant_ave_Ob2_EL)
    
    
    assert(delta_W1.shape == (parameters['W1'].shape[0], L))
    assert(delta_b1.shape == (parameters['b1'].shape[0], 1))
    assert(delta_W2.shape == (1, parameters['W2'].shape[1]))
    # assert(delta_b2.shape == (1, 1))
    
    return delta_W1, delta_b1, delta_W2, delta_b2





### Test

L = 6
# input
X = np.array([-1, -1, -1, 1, 1, 1]).reshape((L,1))

# Initialize parameters, then retrieve W1, b1, W2, b2. 

parameters = deepnet_FFNN_2.initialize_parameters(L, 2*L)


# Feedforward

Y, cache1, cache2, A1 = deepnet_FFNN_2.FFNN_2_model(X, parameters)

# Generate train_set

train_set, train_set_Y = cost_function.markov_chain(X, Y, parameters, n_sample = 5)

# Backpropagation

delta_W1, delta_b1, delta_W2, delta_b2 = compute_gradient(train_set, parameters)

# print('w1:',delta_W1,'b1:', delta_b1, 'w2:',delta_W2, 'b2:',delta_b2)
