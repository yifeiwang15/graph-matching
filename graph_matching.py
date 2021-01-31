import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

def heuristic(M, A, I):
    '''
    Make a soft assignment matrix to a permutation matrix.
    Due to some rules.
    We just set the maximum element in each column to 1 and
    all others to 0.
    This heuristic will always return a permutation matrix
    from a row dominant doubly stochastic matrix.
    '''
    M = normalize(M, norm='l2',axis=1)*normalize(M, norm='l2',axis=1)
    for i in range(A+1):
        index = np.argmax(M[i,:]) # Get the maximum index of each row
        M[i,:] = 0
        if index != I-1:
            M[:,index] = 0
        M[i,index] = 1
    M = M[:A,:I]
    return M


def compatibility(atr1, atr2):
    #Consider the order to return 0 or inf

    if np.isnan(atr1).any() or np.isnan(atr1).any():
        return 0
    if (atr1 == float('Inf')).any() or (atr2 == float('Inf')).any():
        return float('Inf')
    if len(atr1) != len(atr2):
        return 0
    if (atr1 == 0).all() or (atr2 == 0).all():
        return 0


    dim = len(atr1)
    score = np.exp(-((atr1-atr2)**2).sum()/2)/(np.sqrt(2*np.pi)**dim)
    #score = atr1 * atr2
    return score

def compatibility_binary(atr1, atr2):
    # Consider the order to return 0 or inf
    # For binary features.

    if np.isnan(atr1).any() or np.isnan(atr1).any():
        return 0
    if (atr1 == float('Inf')).any() or (atr2 == float('Inf')).any():
        return float('Inf')
    if len(atr1) != len(atr2):
        return 0
    if (atr1 == 0).all() or (atr2 == 0).all():
        return 0


    dim = len(atr1)

    score = (atr1 * atr2).sum()
    return score

def similarity(a, b):
    sigma = 1
    return np.exp(-np.linalg.norm(a - b) / sigma) # e^(norm(a-b)/sigma)

def pre_compute_compatibility(ARG1, ARG2, alpha=1, stochastic=0, node_binary=True, edge_binary=True):
    '''
    Compute the best matching with two ARGs.

    '''

    beta_0 = 0.1
    threshold = 0.1

    # Size of the real match-in matrix
    A = ARG1.g.number_of_nodes()
    I = ARG2.g.number_of_nodes()
    real_size = [A, I] # ???
    augment_size = [A+1, I+1] # size of the matrix with slacks


    # initialize beta to beta_0
    beta = beta_0

    #### compute c_aibj ####
    # nil node compatibility percentage
    prct = 10

    ## pre-calculate the node compatibility
    C_n = np.zeros(augment_size)

    if node_binary:
        C_n[:A,:I] = cdist(ARG1.node_vector(), ARG2.node_vector(), compatibility_binary)
    else:
        C_n[:A,:I] = cdist(ARG1.node_vector(), ARG2.node_vector(), compatibility)

    # Add score to slacks
    C_n[A,:-1] =  np.percentile(C_n[:A,:I],prct,0)
    C_n[:-1,I] =  np.percentile(C_n[:A,:I],prct,1)
    C_n[A,I] = 0

    # times the alpha weight
    C_n = alpha*C_n

    ## pre-calculate the edge compatibility
    C_e = dict()
    for (key_a, key_b) in list(ARG1.g.edges):
        for (key_i, key_j) in list(ARG2.g.edges):
            if edge_binary:
                C_e[(key_a, key_b, key_i, key_j)] = similarity(ARG1.g.edges[key_a, key_b]['eattr'],
                                                               ARG2.g.edges[key_i, key_j]['eattr'])
            else:
                C_e[(key_a, key_b, key_i, key_j)] = similarity(ARG1.g.edges[key_a, key_b]['eattr'],
                                                               ARG2.g.edges[key_i, key_j]['eattr'])
    #TODO set to 0.1 or INF? by original code, there can be some changes
    for i in range(A + 1):
        C_e[i, A] = threshold
        C_e[A, i] = threshold

    return C_n, C_e

# graph matching algorithm!

def graph_matching(C_n, C_e, ARG1, ARG2, beta_0=0.1, beta_f=20, beta_r=1.05,
                   I_0=20, I_1=200, e_B=0.1, e_C=0.01):
    ##  We first do not consider the stochastic.
    # set up the soft assignment matrix
    A = C_n.shape[0] - 1
    I = C_n.shape[1] - 1
    m_Head = np.random.rand(A+1, I+1) # Not an assignment matrix. (Normalized??)
    m_Head[-1,-1] = 0

    # Initialization for parameters

    ## beta is the penalty parameters
    # includes beta_0, beta_f, beta_r

    ## I controls the maximum iteration for each round
    # includes I_0 and I_1

    ## e controls the range
    # includes e_B and e_C

    # begin matching
    beta = beta_0

    stochastic = False ### we first do not consider this case

    while beta < beta_f:

        ## Round-B
        #check if converges
        converge_B = False
        I_B = 0
        while (not converge_B) and I_B <= I_0: # Do B until B is converge or iteration exceeds
            if stochastic:
                m_Head = m_Head ### + ???

            old_B = m_Head # the old matrix
            I_B += 1

            # Build the partial derivative matrix Q
            Q = np.zeros([A+1, I+1])

            # Edge attribute
            for (key_a, key_b) in list(ARG1.g.edges):
                for (key_i, key_j) in list(ARG2.g.edges):
                        Q[key_a,key_i] += C_e[(key_a, key_b, key_i, key_j)] * m_Head[key_b, key_j]
                        # print(C_e[(key_a, key_b, key_i, key_j)] , m_Head[key_a, key_i])
            # Node attribute
            Q = Q + C_n
            # Update m_Head
            m_Head = np.exp(beta*Q)
            m_Head[-1, -1] = 0

            converge_C = False
            I_C = 0
            m_Head = normalize(m_Head, norm='l2',axis=0)*normalize(m_Head, norm='l2',axis=0)                # By row
            m_Head = normalize(m_Head, norm='l2',axis=1)*normalize(m_Head, norm='l2',axis=1)
            while (not converge_C) and I_C <= I_1: # Do C until C is converge or iteration exceeds
                I_C += 1
                old_C = m_Head

                # Begin alternative normalization.
                # Do not consider the row or column of slacks
                # by column
                m_Head = normalize(m_Head, norm='l2',axis=0)*normalize(m_Head, norm='l2',axis=0)
                # By row
                m_Head = normalize(m_Head, norm='l2',axis=1)*normalize(m_Head, norm='l2',axis=1)

                # print(sum(m_Head))
                # update converge_C
                converge_C = abs(sum(sum(m_Head-old_C))) < e_C

            # update converge_B
            converge_B = abs(sum(sum(m_Head[:A,:I]-old_B[:A,:I]))) < e_B
            # print(converge_B, abs(sum(sum(m_Head[:A,:I]-old_B[:A,:I]))))
        # update beta
        beta *= beta_r
        # print(beta, beta_f, beta_r)
    match_matrix = heuristic(m_Head, A, I)
    #match_matrix = m_Head
    return match_matrix

def match_score(match_matrix, idx1, idx2):
    match1 = [idx1[i] for i in match_matrix.nonzero()[0]]
    match2 = [idx2[i] for i in match_matrix.nonzero()[1]]
    score = 0
    for i in range(len(match1)):
        if match1[i] == match2[i]:
            score += 1
    return score/len(match1)
