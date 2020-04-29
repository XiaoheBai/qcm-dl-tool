import numpy as np
import deepnet
import cost_function
import gradient_descent
import grad_check
import matplotlib.pyplot as plt


### Compute the S matrix

def compute_S(train_set, parameters):
    """
    Compute the S matrix for Stochastic Reconfiguration
    The algorithm is similar to compute f 

    Arguments:
    train_set -- Training set with each row representing one |sigma>
    parameters -- python dictionary containing your parameters: Wl, bl

    Returns:
    S -- the S matrix for Stochastic Reconfiguration
    """

    n_sample = len(train_set)

    O_thetas = {} # Python dictionary to store all O_theta for each spin_configuration

    for i in range(len(train_set)):
        spin_configuration = train_set[i].reshape((-1,1))
        Y, Z = deepnet.FFNN_paper_model(spin_configuration, parameters)
        O_W1, O_b1 = gradient_descent.compute_O_operator(spin_configuration, Z, parameters)
        O_theta = grad_check.O_gradients_to_vector(O_W1, O_b1)
        # print('O_theta shape:', O_theta.shape)
        O_thetas['sigma'+str(i)] = O_theta

    K = O_thetas['sigma0'].shape[0]
    # # Compute S_check elementwise
    """ This is only for check. It should not be used in iteration as it's too slow. """

    # S_check = np.zeros((K,K), dtype=complex)

    # for k in range(K):
    #     for k_prime in range(K):

    #         first_term = 0
    #         second_term = 0
    #         third_term = 0

    #         for i in range(n_sample):
    #             first_term += np.conjugate(O_thetas['sigma'+str(i)].item(k)) * O_thetas['sigma'+str(i)].item(k_prime)
    #             second_term += np.conjugate(O_thetas['sigma'+str(i)].item(k))
    #             third_term += O_thetas['sigma'+str(i)].item(k_prime)

    #         S_check.itemset((k,k_prime), first_term/n_sample - second_term / n_sample * third_term / n_sample)

    
    # Compute S using algorithm to speed up
    quant_ave_O_star_O = np.zeros((K, K), dtype=complex)
    quant_ave_O = np.zeros((K, 1), dtype=complex)

    for i in range(n_sample):
        quant_ave_O_star_O = quant_ave_O_star_O + np.dot(np.conjugate(O_thetas['sigma'+str(i)]), O_thetas['sigma'+str(i)].T)
        quant_ave_O = quant_ave_O + O_thetas['sigma'+str(i)]
    
    S = quant_ave_O_star_O / n_sample - np.dot(np.conjugate(quant_ave_O), quant_ave_O.T) / (n_sample**2)

    return S




### Regularize S matrix

def regularize_S_matrix(S_matrix, epsilon):
    """ Regularize S matrix """
    S_matrix[S_matrix<0] = 0

    regularization = np.zeros((S_matrix.shape[0], S_matrix.shape[1]))
    np.fill_diagonal(regularization, epsilon)
    
    S_matrix_reg = S_matrix + regularization
    return S_matrix_reg




### Update Parameters

def sr_update_parameters(parameters, S_matrix_reg, f, learning_rate):
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
    L = parameters['W1'].shape[1]
   
    new_theta = grad_check.dictionary_to_vector(parameters) - learning_rate * np.dot(np.linalg.pinv(S_matrix_reg) , grad_check.f_to_vector(f) )
    
    parameters = grad_check.vector_to_dictionary(new_theta, L)
    
    return parameters



### Test
if __name__ == '__main__':

    L = 6
    # input
    X = np.array([-1,  1,  -1, 1, -1, 1]).reshape((L,1))

    # Initialize parameters, then retrieve W1, b1, W2, b2. 
    parameters = deepnet.initialize_parameters(L, n_h = 2*L, seed=1234, sigma=0.01)
    
    # Y, Z = deepnet.FFNN_paper_model(X, parameters)

    # train_set, train_set_Y = cost_function.markov_chain(X, Y, parameters, n_sample = 5)

    # S = compute_S(train_set, parameters)
    # print("S_check:", S_check)
    

    costs = []
    num_iterations = 600


    for iter in range(0, num_iterations):
        # Feedforward

        Y, Z = deepnet.FFNN_paper_model(X, parameters)

        # Generate train_set

        train_set, train_set_Y = cost_function.markov_chain(X, Y, parameters, n_sample = 1000)

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        """ Finished above and in the process of generating the train_set """
            
        # Compute cost
        hamiltonian, train_set_h_local = cost_function.compute_hamiltonian(train_set, train_set_Y, parameters)

        # Compute f
        f = gradient_descent.compute_f(train_set, train_set_h_local, parameters, hamiltonian)
        
        # Compute S
        S = compute_S(train_set, parameters)
            
        # Update parameters.
        parameters = sr_update_parameters(parameters, S, f, 0.003)
                                            
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


