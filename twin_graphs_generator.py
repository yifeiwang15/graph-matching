import numpy as np
import math
import random

def generate_twin_graphs(size=100, weight_range=1, connected_rate=0.2, noise_rate=0.2,
                                      edge_feature_dim=3):
     '''
     G is a template matrix for pattern.
     Generate two twin graphs G1, G2 from G for graph matching.
     (The numbers of nodes in G1 and G2 may be different from G)
     Randomly permute nodes and add some noise to edges and nodes.

     Both of edge weights and node attributes can be negative.

     size: number of nodes in G.
     weight_range: default:1, for scale
     '''
     # Create G
     # M = np.triu((np.random.rand(size,size)*2-1)*weight_range, 1) # upper tri matrix. Diag elements are 0.
     # Mi = np.triu(np.random.randint(edge_feature_dim, size=(size, size)),1) # M features type index
     connected_nodes = np.triu(np.random.rand(size,size) < connected_rate, 1).astype(int)
     # M_tmp = np.zeros([size, size, edge_feature_dim])
     M_tmp = {}
     for i in range(size):
         for j in range(size):
             if connected_nodes[i, j]:
                 # M_tmp[i, j, Mi[i, j]] = M[i, j]
                 edge_feature = np.random.rand(edge_feature_dim) * weight_range # single edge feature vector
                 M_tmp[(i, j)] = edge_feature
                 M_tmp[(j, i)] = edge_feature # make M_tmp symetric matrix
     M = M_tmp
     # M = M + np.transpose(M, axes=[1, 0, 2])
     M1 = dict()
     M2 = dict()
     M1_size = round((1+random.random()*noise_rate)*size)
     M2_size = round((1+random.random()*noise_rate)*size)

     idx1 = np.random.permutation(M1_size)
     idx2 = np.random.permutation(M2_size)

     for (i, j) in M.keys():
         M1[(idx1[i], idx1[j])] = M[(i, j)]
         M2[(idx2[i], idx2[j])] = M[(i, j)]
     # Generate the permutation of G1 and G2
     # M1 = np.triu((np.random.rand(M1_size,M1_size)*2-1)*weight_range, 1) # upper tri matrix. Diag elements are 0.
     # M1i = np.triu(np.random.randint(edge_feature_dim, size=(M1_size, M1_size)),1) # M1 features type index
     connected_nodes_m1 = np.triu(np.random.rand(M1_size,M1_size) < connected_rate, 1).astype(int)
     connected_nodes_m1[:size, :size] = connected_nodes
     # M1_tmp = np.zeros([size, size, edge_feature_dim])
     for i in range(M1_size):
         for j in range(M1_size):
             if connected_nodes_m1[i, j] and (i, j) not in M1.keys():
                 # M1_tmp[i, j, M1i[i, j]] = M1[i, j]
                 edge_feature = np.random.rand(edge_feature_dim) * weight_range # single edge feature vector
                 M1[(i, j)] = edge_feature
                 M1[(j, i)] = edge_feature # make M_tmp symetric matrix
     # M1 = M1 + np.transpose(M1, axes=[1, 0, 2])

     # M2 = np.triu((np.random.rand(M2_size,M2_size)*2-1)*weight_range, 1) # upper tri matrix. Diag elements are 0.
     # M2i = np.triu(np.random.randint(edge_feature_dim, size=(M2_size, M2_size)),1) # M2 features type index
     connected_nodes_m2 = np.triu(np.random.rand(M2_size,M2_size) < connected_rate, 1).astype(int)
     connected_nodes_m2[:size, :size] = connected_nodes
     # M2_tmp = np.zeros([size, size, edge_feature_dim])
     for i in range(M2_size):
         for j in range(M2_size):
             if connected_nodes_m2[i, j] and (i, j) not in M2.keys():
                 # M2_tmp[i, j, M2i[i, j]] = M2[i, j]
                 edge_feature = np.random.rand(edge_feature_dim) * weight_range # single edge feature vector
                 M2[(i, j)] = edge_feature
                 M2[(j, i)] = edge_feature # make M_tmp symetric matrix
     # M2 = M2 + np.transpose(M2, axes=[1, 0, 2])


     # node attributes
     V = (np.random.rand(size)*2-1)*weight_range
     V1 = (np.random.rand(M1_size)*2-1)*weight_range
     V2 = (np.random.rand(M2_size)*2-1)*weight_range
     V1[:size] = V
     V2[:size] = V

     # Generate random permutation matrix

     V1 = V1[idx1]
     V2 = V2[idx2]


     #The original method for generating noises.
     # Adding noise
     # Adding noise to nodes
     V1_noise = np.random.normal(0,1,M1_size)
     V1_noise = V1_noise*weight_range*noise_rate
     V1 = V1 + V1_noise
     V2_noise = np.random.normal(0,1,M2_size)
     V2_noise = V2_noise*weight_range*noise_rate
     V2 = V2 + V2_noise

     # Adding noise to edges
     # The disturbed graph will be also undirected.(Different from the matlab version)

     # M1_noise = np.triu(np.random.normal(0,1,[M1_size,M1_size]), 1)
     # M1_noise = M1_noise*weight_range*noise_rate
     # M1_noise = M1_noise + M1_noise.transpose()
     # none_zero = M1.nonzero()
     # M1[none_zero] += M1_noise[none_zero]
     for i in range(M1_size):
         for j in range(M1_size):
             if connected_nodes_m1[i, j]:
                 # M1[i, j, Mi[i, j]] += M1_noise[i, j]
                 noise = np.random.normal(0, 1, 1)
                 M1[(i, j)] += noise
                 M1[(j, i)] += noise

     # M2_noise = np.triu(np.random.normal(0,1,[M2_size,M2_size]), 1)
     # M2_noise = M2_noise*weight_range*noise_rate
     # M2_noise = M2_noise + M2_noise.transpose()
     # none_zero = M1.nonzero()
     # M1[none_zero] += M1_noise[none_zero]
     for i in range(M2_size):
         for j in range(M2_size):
             if connected_nodes_m2[i, j]:
                 # M2[i, j, Mi[i, j]] += M2_noise[i, j]
                 noise = np.random.normal(0, 1, 1)
                 M2[(i, j)] += noise
                 M2[(j, i)] += noise
     '''
     # The second method for generating noises.
     M1_noise = np.triu(np.random.rand(M1_size, M1_size), 1)
     M1_noise = M1_noise + M1_noise.transpose()
     M1_noise = (M1_noise >= noise_rate).astype(int)
     M1 = M1*M1_noise

     M2_noise = np.triu(np.random.rand(M2_size, M2_size), 1)
     M2_noise = M2_noise + M2_noise.transpose()
     M2_noise = (M2_noise >= noise_rate).astype(int)
     M2 = M2*M2_noise
     '''
     # try return M1, M2, V1, V2,
     # also return idx1, idx2 (The permutation)
     return M1, M2, V1, V2, idx1, idx2

generate_twin_graphs()
