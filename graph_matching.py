import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from twin_graphs_generator import generate_twin_graphs
from ARG import ARG



class GraphMatching(object):
    '''
    The implementation of a graduate assignment graph matching algorithm integrated with
    networkx.
    To use:
        >>> algorithm = GraphMatching(size=0, weight_range=0, connected_rate=0, noise_rate=0,
                                      ARG1=G1, ARG2=G2, idx1=idx1, idx2=idx2)
        >>> match_matrix = algorithm.graph_matching()
        >>> final_score, match1, match2 = GraphMatching.match_score(match_matrix, algorithm.idx1, algorithm.idx2)
    '''

    def __init__(self, size, weight_range, connected_rate, noise_rate,
                 beta_0=20, beta_f=201, beta_r=1.25, I_0=20, I_1=200, e_B=0.1, e_C=0.01,
                 ARG1=None, ARG2=None, idx1=None, idx2=None):

        if (ARG1 is None) ^ (ARG2 is None):
            raise Exception('both ARG needed to be passed in as parameters')
        if ARG1 is None:
            M1, M2, V1, V2, idx1, idx2 = generate_twin_graphs(size=size,
                                                          weight_range=weight_range,
                                                          connected_rate=connected_rate,
                                                          noise_rate=noise_rate)
            print('Randomly generated ARG1, ARG2')
            print(M1.keys(), M2.keys(), V1, V2, idx1, idx2)

            self.ARG1 = ARG()
            self.ARG1.from_matrix(M=M1, V=V1)
            self.ARG2 = ARG()
            self.ARG2.from_matrix(M=M2, V=V2)
        else:
            print('Passed in ARG1, ARG2')
            self.ARG1 = ARG1
            self.ARG2 = ARG2

        C_n, C_e = self.pre_compute_compatibility(alpha=1, node_binary=False)
        self.C_n = C_n
        self.C_e = C_e

        self.idx1 = idx1
        self.idx2 = idx2

        # set up the algorithm parameters
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self.beta_r = beta_r
        self.I_0 = I_0
        self.I_1 = I_1
        self.e_B = e_B
        self.e_C = e_C



    def pre_compute_compatibility(self, alpha=1, node_binary=True):
        '''
        Compute the best matching with two ARGs.
        ARGS:
            alpha: weight of node attributes
            node_binary: True if the node attribute is binary
        '''

        # Size of the real match-in matrix
        A = self.ARG1.number_of_nodes()
        I = self.ARG2.number_of_nodes()
        augment_size = [A + 1, I + 1]  # size of the matrix with slacks

        # nil node compatibility percentage
        prct = 10

        # pre-calculate the node compatibility
        C_n = np.zeros(augment_size)

        if node_binary:
            C_n[:A, :I] = cdist(self.ARG1.nodes_vectors(), self.ARG2.nodes_vectors(),
                                GraphMatching.compatibility_binary)
        else:
            C_n[:A, :I] = cdist(self.ARG1.nodes_vectors(), self.ARG2.nodes_vectors(), GraphMatching.compatibility)


        # Add score to slacks
        C_n[A, :-1] = np.percentile(C_n[:A, :I], prct, 0)
        C_n[:-1, I] = np.percentile(C_n[:A, :I], prct, 1)
        C_n[A, I] = np.mean(np.sum(C_n[A, :-1]) + np.mean(C_n[:-1, I]))

        # times the alpha weight
        C_n = alpha * C_n

        ## pre-calculate the edge compatibility
        C_e = dict()
        for (key_a, key_b) in list(self.ARG1.edges):
            for (key_i, key_j) in list(self.ARG2.edges):
                C_e[(key_a, key_b, key_i, key_j)] = GraphMatching.similarity(self.ARG1.edges[key_a, key_b]['eattr'],
                                                                             self.ARG2.edges[key_i, key_j]['eattr'])
        return C_n, C_e

    def graph_matching(self):
        # set up the soft assignment matrix
        A = self.C_n.shape[0] - 1
        I = self.C_n.shape[1] - 1
        m_Head = np.ones(shape=(A + 1, I + 1))
        m_Head[-1, -1] = 0

        # begin matching
        beta = self.beta_0

        while beta < self.beta_f:

            ## Round-B
            # check if converges
            converge_B = False
            I_B = 0
            while (not converge_B) and I_B <= self.I_0:  # Do B until B is converge or iteration exceeds

                old_B = m_Head # the old matrix
                I_B += 1

                # Build the partial derivative matrix Q
                Q = np.zeros([A + 1, I + 1])

                # Edge attribute
                # Notice in networkx implementation key_a will be always less than Key_b
                # so we need to permute key_a and key_b in the loop (same for key_i, key_j)
                for (key_a, key_b) in list(self.ARG1.edges):
                    for (key_i, key_j) in list(self.ARG2.edges):
                        Q[key_a, key_i] += self.C_e[(key_a, key_b, key_i, key_j)] * m_Head[key_b, key_j]
                        Q[key_b, key_i] += self.C_e[(key_a, key_b, key_i, key_j)] * m_Head[key_a, key_j]

                        Q[key_a, key_j] += self.C_e[(key_a, key_b, key_i, key_j)] * m_Head[key_b, key_i]
                        Q[key_b, key_j] += self.C_e[(key_a, key_b, key_i, key_j)] * m_Head[key_a, key_i]

                # Node attribute
                Q = Q + self.C_n
                # Update m_Head
                m_Head = np.exp(beta * Q)
                # m_Head[-1, -1] = 0

                converge_C = False
                I_C = 0

                while (not converge_C) and I_C <= self.I_1:  # Do C until C is converge or iteration exceeds
                    I_C += 1
                    old_C = m_Head

                    # Begin alternative normalization.
                    # Do not consider the row or column of slacks
                    # by row, avoid normalize with m_Head[-1, -1]
                    m_Head = normalize(m_Head, norm='l1', axis=0)
                    # By row
                    m_Head = normalize(m_Head, norm='l1', axis=1)

                    # update converge_C
                    converge_C = np.sum(abs(m_Head - old_C)) < self.e_C

                # update converge_B
                converge_B = np.sum(abs(m_Head[:A, :I] - old_B[:A, :I])) < self.e_B
                # print('converge_B', np.sum(abs(m_Head[:A, :I] - old_B[:A, :I])) )
            # update beta
            beta *= self.beta_r
            # print(beta, beta_f, beta_r)
        print(m_Head)
        match_matrix = GraphMatching.heuristic(m_Head, A, I)
        # match_matrix = m_Head
        return match_matrix

    @staticmethod
    def heuristic(M, A, I):
        '''
        Make a soft assignment matrix to a permutation matrix.
        Due to some rules.
        We just set the maximum element in each column to 1 and
        all others to 0.
        This heuristic will always return a permutation matrix
        from a row dominant doubly stochastic matrix.
        '''
        # M = normalize(M, norm='l2', axis=1) * normalize(M, norm='l2', axis=1)
        for i in range(A): # skip slack row
            index = np.argmax(M[i, :])  # Get the maximum index of each row
            M[i, :] = 0
            if index != I: # skip slack column
                M[:, index] = 0
            M[i, index] = 1
        M = M[:A, :I]
        return M

    @staticmethod
    def compatibility(atr1, atr2):
        # Consider the order to return 0 or inf

        if np.isnan(atr1).any() or np.isnan(atr1).any():
            return 0
        if (atr1 == float('Inf')).any() or (atr2 == float('Inf')).any():
            return float('Inf')
        if len(atr1) != len(atr2):
            return 0
        if (atr1 == 0).all() or (atr2 == 0).all():
            return 0

        dim = len(atr1)
        score = np.exp(-((atr1 - atr2) ** 2).sum() / 2) / (np.sqrt(2 * np.pi) ** dim)
        #e^((sum((a-b)^2) / 2) / sqrt(2*pi)^dim)
        return score

    @staticmethod
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

    @staticmethod
    def similarity(a, b):
        sigma = 1
        return np.exp(-np.linalg.norm(a - b) / sigma)  # e^(norm(a-b)/sigma)

    # graph matching algorithm!
    @staticmethod
    def match_score(match_matrix, idx1, idx2):
        '''
        calculate the percentage of correctly matched nodes
        :param match_matrix: matching result with slack variables
        :param idx1: permutation of nodes from original graph to graph1
        :param idx2: permutation of nodes from original graph to graph2
        :return: the percentage,
                 matching array of nodes from graph1 to original graph
                 matching array of nodes from graph2 to original graph
        '''
        g1idx_to_gvalue = np.zeros(len(idx1))
        for i, v in enumerate(idx1):
            g1idx_to_gvalue[v] = i

        g2idx_to_gvalue = np.zeros(len(idx2))
        for i, v in enumerate(idx2):
            g2idx_to_gvalue[v] = i

        match1 = [i for i in match_matrix.nonzero()[0]]
        match2 = [i for i in match_matrix.nonzero()[1]]

        score = 0
        for i in range(len(match1)):
            if g1idx_to_gvalue[match1[i]] == g2idx_to_gvalue[match2[i]]:
                score += 1
        return score / len(idx1), match1, match2
