import numpy as np
import deepnet
import cost_function
import gradient_descent
import grad_check
import matplotlib.pyplot as plt

### Compute the S matrix

def compute_S_matrix(train_set, parameters):
    """
    Compute the S matrix for Stochastic Reconfiguration

    Arguments:
    train_set -- train_set -- Training set with each row representing one |sigma>
    parameters -- python dictionary containing your parameters: Wl, bl

    Returns:
    S_matrix
    """

    n_sample = len(train_set)

    O_thetas = {} # Python dictionary containing all O_theta's: O_thetas['x0'], O_thetas['x1'], ... 

    for i in range(len(train_set)):
        spin_configuration = train_set[i].reshape((-1,1))
        phi, Y, Z = deepnet.FFNN_paper_model(spin_configuration, parameters)
        O_grads = gradient_descent.compute_O_operator(spin_configuration, parameters, Z )
        O_thetas['x' + str(i)] = grad_check.O_gradients_to_vector(O_grads)

        if i == 0:
            O_thetas_average = np.copy(O_thetas['x0'])
        else:
            O_thetas_average = O_thetas_average + O_thetas['x' + str(i)]
        """ The above O_thetas_average has not been divided by n_sample"""
    
    O_thetas_average = O_thetas_average / n_sample
    
    for i in range(len(train_set)):
        O_diff = O_thetas['x'+str(i)] - O_thetas_average
        
        if i == 0:
            S_matrix = np.copy( np.dot(O_diff, O_diff.T) )
        else:
            S_matrix = S_matrix + ( np.dot( O_diff, O_diff.T) )
        """ The above S matrix has not been divided by n_sample"""

    S_matrix = np.real( S_matrix / n_sample )

    assert(S_matrix.shape == (O_thetas_average.shape[0], O_thetas_average.shape[0]))
    return S_matrix




### Regularize S matrix

def regularize_S_matrix(S_matrix, epsilon):
    """ Regularize S matrix """
    S_matrix[S_matrix<0] = 0

    regularization = np.zeros((S_matrix.shape[0], S_matrix.shape[1]))
    np.fill_diagonal(regularization, epsilon)
    
    S_matrix_reg = S_matrix + regularization
    return S_matrix_reg




### Update Parameters

def sr_update_parameters(parameters, S_matrix_reg, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
   
    new_theta = grad_check.dictionary_to_vector(parameters)[0] - learning_rate * np.dot(np.linalg.pinv(S_matrix_reg) , grad_check.gradients_to_vector(grads) )
    
    parameters = grad_check.vector_to_dictionary(new_theta)
    
    return parameters



### Test

L = 6
# input
X = np.array([-1,  1,  -1, 1, -1, 1]).reshape((L,1))

# Initialize parameters, then retrieve W1, b1, W2, b2. 

parameters = deepnet.initialize_parameters(L, n_h = 2*L, seed=1234, sigma=0.01)

costs = []
num_iterations = 300


for iter in range(0, num_iterations):
    # Feedforward

    phi, Y, Z = deepnet.FFNN_paper_model(X, parameters)

    # Generate train_set

    train_set, train_set_phi = cost_function.markov_chain(X, phi, parameters, n_sample = 1000)

    # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
    """ Finished above and in the process of generating the train_set """
        
    # Compute cost
    hamiltonian = cost_function.compute_hamiltonian(train_set, train_set_phi, parameters)


    # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
    grads = gradient_descent.compute_gradients(train_set, parameters)
    S_matrix = compute_S_matrix(train_set, parameters)
    S_matrix_reg = regularize_S_matrix(S_matrix, epsilon = 0.00001)
        
    # Update parameters.
    parameters = sr_update_parameters(parameters, S_matrix_reg, grads, 0.001)
                                           
    # Retrieve W1, b1, W2, b2 from parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    # W2 = parameters["W2"]
    # b2 = parameters["b2"]

    # Print the cost
    print("Cost after iteration {}: {}".format(iter, np.squeeze(hamiltonian)))
    costs.append(hamiltonian)
    

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(0.005))
plt.show()


