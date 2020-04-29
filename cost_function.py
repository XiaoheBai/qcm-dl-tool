import numpy as np
import deepnet 
import random


### Compute the loss function

# Compute the local Hamiltonian h_local

def compute_h_local(X, Y, parameters, J1 = 1, J2 = 0.4):
    """ 
    Only valid for Heisenberg J1-J2 model:
    H = J1 * sum (sigma_i * sigma_i+1) + J2 * sum (sigma_i * sigma_i+2)

    Argument:
    X -- |sigma>
    Y -- Y(|sigma>, parameters)
    parameters -- python dictionary containing all parameters
    J1 -- J1 parameter in J1-J2 Heisenberg model
    J2 -- J2 parameter in J1-J2 Heisenberg model   
    
    Returns:
    h_local -- Local Hamiltonian of the input |sigma> X
    """

    L = X.shape[0]

    h_local = 0

    for i in range(L):
        """ consider the nearst neighbour"""
        if  X.item(i) *  X.item( (i+1)%  L ) == 1:
            """ This case includes (1,1) and (-1,-1)"""
            h_local += J1

        if  X.item(i) *  X.item( (i+1)%  L ) == -1:
            """ This case includes (1,-1) and (-1,1)"""
            new_X =  np.copy(X)
            new_X[[i, (i+1)%  L]] = new_X[[(i+1)%  L, i]]
            new_Y = deepnet.FFNN_paper_model(new_X, parameters)[0]
            h_local += J1 * ( 2 * np.exp(new_Y - Y) -1)
            
        
        """consider the next nearst neighbour"""
        if X.item(i) *  X.item( (i+2)%  L ) == 1:
            h_local += J2
        
        if X.item(i) *  X.item( (i+2)%  L ) == -1:
            new_X =  np.copy(X)
            new_X[[i, (i+2)%  L]] = new_X[[(i+2)%  L, i]]
            new_Y = deepnet.FFNN_paper_model(new_X, parameters)[0]
            h_local += J2 * ( 2 * np.exp(new_Y - Y) -1)
            
        
    return  h_local


# Generate the Markov chain using importance sampling

def markov_chain(X, Y, parameters, n_sample = 1000):
    """
    Argument:
    X -- |sigma_0>
    Y -- Y(|sigma_0>, parameters)
    parameters -- python dictionary containing all parameters (output of initialization function)
    n_sample -- number of train set (length of the markov chain)
    
    Returns:
    train_set -- Training set with each row representing one |sigma>
    train_set_Y -- The deepnet output Y of each spin configuration in train_set
    """

    L = X.shape[0]

    train_set = np.copy(X.T)
    # print(train_set)
    train_set_Y = [Y]

    for n in range(1,n_sample):
        
        # Copy last spin configuration
        propose_X = np.copy( train_set[n-1] )
        # print(propose_X)

        """ Generate random int i-th position"""
        i = random.randint(0, L-1)

        """ Generate the random distance: dmax = 2 """
        distance = random.randint(-2, 2)

        j = (i+distance)%L
        
        propose_X[[i,j]] = propose_X[[j,i]]
        propose_X = propose_X.reshape((1,L))
        # print(propose_X)

        propose_Y = deepnet.FFNN_paper_model(propose_X.T, parameters)[0]

        """ Accept / Reject """
        ratio = lambda propose_Y, Y: np.exp(2) if (propose_Y.real - Y.real)>=1. else np.exp(2 * (propose_Y.real - Y.real))
        r = min(1 , ratio(propose_Y, Y))
        # print(propose_X, propose_Y, train_set_Y[n-1], r)
        random_number = random.uniform(0, 1)

        if random_number > r: # rejected
            train_set = np.r_[train_set, train_set[n-1].reshape((1,L))]
            train_set_Y.append(train_set_Y[n-1])
        
        elif random_number <= r: # accept
            train_set = np.r_[train_set, propose_X]
            train_set_Y.append(propose_Y)
        
        # print(n,"-th iteration: train set:", train_set[-1], train_set_Y[-1])

    return train_set, train_set_Y

# Compute the loss function: Hamiltonian

def compute_hamiltonian(train_set, train_set_Y, parameters):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    train_set -- Training set with each row representing one |sigma>
    train_set_Y -- The deepnet output Y of each spin configuration in train_set
    parameters -- python dictionary containing all parameters (output of initialization function)
    
    Returns:
    hamiltonian -- Quantum Average of the Hamiltonian of the system (cost function of neural network)
    train_set_h_local -- h_local list tracked for each |sigma> in train_set, stored for computing the gradient descent efficiently
    """
    
    n_sample = len(train_set)

    L = len(train_set[0])

    hamiltonian = 0
    train_set_h_local = []

    for i in range(n_sample):
        spin_configuration = train_set[i].reshape((L,1))
        Y = train_set_Y[i]
        train_set_h_local.append(compute_h_local(spin_configuration, Y, parameters, J1 = 1, J2 = 0.4))
        
    hamiltonian = sum(train_set_h_local) /  n_sample    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    # hamiltonian = hamiltonian.real

    assert(hamiltonian.shape == ())
    assert(len(train_set_h_local) == n_sample)
    
    return hamiltonian, train_set_h_local





### Test

if __name__ == '__main__':
    L = 6

    # input
    X = np.array([-1, -1, 1, -1, 1, 1]).reshape((L,1))

    # Initialize parameters, then retrieve W1, b1. 
    parameters = deepnet.initialize_parameters(L, n_h = 2*L, seed=1234, sigma=0.01)
    # print(parameters)


    # Feedforward
    Y, Z = deepnet.FFNN_paper_model(X, parameters)
    # print(Y)

    # Try h_local function
    # h_local = compute_h_local(X, Y, parameters, J1 = 1, J2 = 0.4)
    # print(h_local)


    # generate train_set
    train_set, train_set_Y = markov_chain(X, Y, parameters, n_sample = 1000)
    # print(len(train_set))
    # print(train_set_phi[0].shape)

    # compute loss
    hamiltonian, train_set_h_local = compute_hamiltonian(train_set, train_set_Y, parameters)
    print(hamiltonian)