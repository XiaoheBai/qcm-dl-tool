import numpy as np
import deepnet
import cost_function
import gradient_descent


### Reshape parameters and gradients to a big vector

# Reshape parameters
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    Wl -- weight matrix of the visible (0-th) layer, shape (layer_dims[0], L)
                    bl -- bias matrix of the visible (0-th) layer, shape (layer_dims[0], 1)
    
    Returns:
    theta -- a big vector containing all parameters with the sequence: 
                    W1[1], W1[2], ..., b1, W2[1], ... b2, ...
    keys -- Information about the sequence in 'theta'
    """
    keys = []
    count = 0
    for key in ["W1", "b1"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    
    return theta, keys

# Reshape the tiny shifted big vector theta back to the parameters dictionary
def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    
    Arguments: 
    theta -- a big vector containing all parameters with the sequence: 
                    W1[1], W1[2], ..., b1, W2[1], ... b2, ...

    Returns:
    parameters -- python dictionary containing your parameters:
                    Wl -- weight matrix of the visible (0-th) layer, shape (layer_dims[0], L)
                    bl -- bias matrix of the visible (0-th) layer, shape (layer_dims[0], 1)
    """
    parameters = {}
    L = 6
    parameters["W1"] = theta[:2*L**2].reshape((2*L,L))
    parameters["b1"] = theta[2*L**2:].reshape((-1,1))

    return parameters


# Reshape gradients of Y (O_grads) to vector
def O_gradients_to_vector(O_grads):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    
    Arguments:
    O_grads -- python dictionary containing O Operator applied on parameters
                    O_Wl -- Gradient of Y with respect to Wl (matrix of Wl.shape)
                    O_bl -- Gradient of Y with respect to bl (matrix of bl.shape)

    Returns:
    O_theta -- a big vector containing all O_grads with the sequence: 
                    O_W1[1], O_W1[2], ..., O_b1, O_W2[1], ... O_b2, ...
    """
    
    count = 0
    for key in ["O_W1", "O_b1"]:
        # flatten parameter
        new_vector = np.reshape(O_grads[key], (-1,1))
        
        if count == 0:
            O_theta = new_vector
        else:
            O_theta = np.concatenate((O_theta, new_vector), axis=0)
        count = count + 1

    return O_theta

# Reshape gradients of cost function Hamiltonian (grads) to vector
def gradients_to_vector(grads):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    
    Arguments:
    grads -- python dictionary contraining your gradients
                    delta_Wl -- Gradient of the cost with respect to Wl
                    delta_bl -- Gradient of the cost with respect to bl
    
    Returns:
    delta_theta -- a big vector containing all gradients with the sequence: 
                    delta_W1[1], delta_W1[2], ..., delta_b1, delta_W2[1], ... delta_b2, ...
    """
    
    count = 0
    for key in ["delta_W1", "delta_b1"]:
        # flatten parameter
        new_vector = np.reshape(grads[key], (-1,1))
        
        if count == 0:
            delta_theta = new_vector
        else:
            delta_theta = np.concatenate((delta_theta, new_vector), axis=0)
        count = count + 1

    return delta_theta




### Gradient Check

# O gradient check
def O_gradients_check(parameters, O_grads, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation computes correctly the gradient of Y by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "Wl", "bl"
    O_grads -- output of linear_activation_backward, contains O gradients of Y with respect to the parameters. 
    X -- input datapoint, of shape (input size, 1)
    Y -- output of the deepnet
    epsilon -- tiny shift to the input to compute approximated gradient with formula:
                  O_gradapprox = (Y(+) - Y(-))/(2*epsilon)
    
    Returns:
    difference -- difference between the approximated gradient and the backward propagation gradient:
                  || O_grads - O_gradapprox ||_2 / (|| O_grads ||_2 + || O_gradapprox ||_2)
    """
    
    # Set-up variables
    theta, _ = dictionary_to_vector(parameters)
    O_theta = O_gradients_to_vector(O_grads)
    num_parameters = theta.shape[0]
    Y_plus = np.zeros((num_parameters, 1)) # Vector
    Y_minus = np.zeros((num_parameters, 1))
    O_gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute Y_plus[i]. Inputs: "theta, epsilon". Output = "Y_plus[i]".
        thetaplus = np.copy(theta)                   # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                   # Step 2
        Y_plus[i] = deepnet.FFNN_paper_model(X, vector_to_dictionary( thetaplus))[1]     # Step 3
        
        # Compute Y_minus[i]. Inputs: "theta, epsilon". Output = "Y_minus[i]".
        thetaminus = np.copy(theta)                            # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon              # Step 2        
        Y_minus[i] = deepnet.FFNN_paper_model(X, vector_to_dictionary( thetaminus))[1]  # Step 3
        
        # Compute gradapprox[i]
        O_gradapprox[i] = (Y_plus[i] - Y_minus[i]) / (2*epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(O_theta - O_gradapprox)           # Step 1'
    denominator = np.linalg.norm(O_theta) + np.linalg.norm(O_gradapprox)         # Step 2'
    difference = numerator / denominator         # Step 3'

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
 
# Gradients check
def gradients_check(parameters, grads, train_set, train_set_phi, epsilon = 1e-7):
    """
    Checks if the gradient of Hamiltonian computes correctly
    
    Arguments:
    parameters -- python dictionary containing your parameters "Wl", "bl"
    grads -- output of compute_gradients, contains gradients of Hamiltonian with respect to the parameters. 
    train_set -- Training set with each row representing one |sigma>
    train_set_phi -- The wavefunction of each spin configuration in train_set
    epsilon -- tiny shift to the input to compute approximated gradient with formula:
                  O_gradapprox = (H(+) - H(-))/(2*epsilon)
    
    Returns:
    difference -- difference between the approximated gradient and the backward propagation gradient:
                  || grads - gradapprox ||_2 / (|| grads ||_2 + || gradapprox ||_2)
    """
    
    # Set-up variables
    theta, _ = dictionary_to_vector(parameters)
    delta_theta = gradients_to_vector(grads)
    num_parameters = theta.shape[0]
    H_plus = np.zeros((num_parameters, 1)) # Vector
    H_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute H_plus[i]. Inputs: "theta, epsilon". Output = "H_plus[i]".
        thetaplus = np.copy(theta)                   # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                   # Step 2
        H_plus[i] = cost_function.compute_hamiltonian(train_set, train_set_phi, vector_to_dictionary( thetaplus))     # Step 3
        # print('H_plus',i,':',H_plus[i])
        
        # Compute H_minus[i]. Inputs: "theta, epsilon". Output = "H_minus[i]".
        thetaminus = np.copy(theta)                            # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon              # Step 2        
        H_minus[i] = cost_function.compute_hamiltonian(train_set, train_set_phi, vector_to_dictionary( thetaminus))  # Step 3
        # print('H_minus',i,':',H_minus[i])

        # Compute gradapprox[i]
        gradapprox[i] = (H_plus[i] - H_minus[i]) / (2*epsilon)
        # print('gradapprox',i,':', gradapprox[i])
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(delta_theta - gradapprox)           # Step 1'
    denominator = np.linalg.norm(delta_theta) + np.linalg.norm(gradapprox)         # Step 2'
    difference = numerator / denominator         # Step 3'

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

### Test

# L = 6
# # input
# X = np.array([-1, 1, -1, 1, -1, 1]).reshape((L,1))

# # Initialize parameters, then retrieve W1, b1, W2, b2. 

# parameters = deepnet.initialize_parameters(L, [2*L])

# theta, keys = dictionary_to_vector(parameters)

# # print(theta.shape, len(keys) )

# phi, Y, Z = deepnet.FFNN_paper_model(X, parameters)

# # generate train_set
# train_set, train_set_phi = cost_function.markov_chain(X, phi, parameters, n_sample = 10)

# grads = gradient_descent.compute_gradients(train_set, parameters)

# difference = gradients_check(parameters, grads, train_set, train_set_phi, epsilon = 1e-7)

