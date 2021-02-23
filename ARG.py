import numpy as np
import networkx as nx
import random
#test
# There is some problem with class node and class edge.
# (The attribute has error)
# But we only use ARG.nodes_vector and ARG.edges_matrix in this case.


class ARG(nx.Graph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    '''
        Attributed Relational Graphs represents a graph data structure with a list of node.
        M is the edge weight matrix and V is the node attributes.
    '''
    def from_matrix(self, M, V):
        # Check if M is square
        # assert (M.shape == M.transpose().shape), "Input edge weight matrix is not square."
        # Check if numbers of nodes from M and from V is matched.
        # assert (len(M) == len(V)), "Input sizes of edge weight matrix and of nodes attributes do not match."

        # Use dictionary structure to store nodes and edges.
        super(ARG, self).__init__()
        for (a, b), value in M.items():
            self.add_edge(a, b, eattr=value)

        for id in range(len(V)):
            self.add_node(id, nattr=V[id])

        # (TODO) check the dimension and the usage of trainable parameter( especially loss)


    def nodes_vectors(self):
        num_nodes = self.number_of_nodes()
        attribute_name = random.sample(self.nodes[0].keys(), 1)[0]
        attribute_dim = self.nodes[0][attribute_name].shape[0]
        v = np.zeros(shape=(num_nodes, attribute_dim))
        for i in range(num_nodes):
            v[i] = self.nodes[i][attribute_name]
        return v

if __name__ == '__main__':
    g = ARG(M={2:3},V={1})
    ng = nx.Graph()
    ng.add_node(1)
    ng.add_edge(1,3)
    g1 = ARG(ng)
    print(g1.nodes)