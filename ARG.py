import numpy as np
import networkx as nx
#test
# There is some problem with class node and class edge.
# (The attribute has error)
# But we only use ARG.nodes_vector and ARG.edges_matrix in this case.


class ARG(nx.Graph):

    def __init__(self, nx_graph):
        self.update(nx_graph)

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=None, **attr)

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

        # (TODO) check the dimension and the usage of trainnable paramter( especially loss)
        # stand for node weight
        # self.nw =
        # stand for edge weigth
        # self.ew =

    def nodes_vectors(self):
        num_nodes = self.number_of_nodes()
        v = np.zeros(num_nodes)
        for i in range(num_nodes):
            v[i] = self.nodes[i]['nattr']
        return v.reshape([num_nodes,-1])

if __name__ == '__main__':
    g = ARG(M={2:3},V={1})
