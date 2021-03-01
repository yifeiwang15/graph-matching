import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import random
from ARG import ARG
import math
from graph_matching import GraphMatching


def generate_rotation_3Dgraph(n_nodes, rotvect, noise, connection_rate, seed=None):
    '''
    generate a graph with nodes in 3d coordinates and a graph rotate from it
    :param n_nodes: number of nodes
    :param rotvect: rotation vector axis:[x, y, z]
    :param noise: noise to nodes value
    :param connection_rate: possibility of having a edge between two nodes
    :param seed: random seed
    :return: graph1, rotation of graph1, permutation of graph1, permutation of rotated graph1
    '''
    if seed is not None:
        random.seed(seed)

    r = R.from_rotvec(rotvect)#[0, 0, np.pi / 2])
    # np.matmul(r.as_matrix(), np.array([1, 1, 1]))
    # Generate a dict of positions
    G1 = nx.Graph()
    pos1 = {i: np.random.rand(3) for i in range(n_nodes)}
    for i in range(n_nodes):
        G1.add_node(i, pos=pos1[i])
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v and random.random() <= connection_rate:
                dist= G1.nodes[u]['pos'] - G1.nodes[v]['pos']
                G1.add_edge(u, v, eattr=np.linalg.norm(np.array(dist)))

    G2 = nx.Graph()
    n_nodes_r = (1 + noise) * n_nodes

    idx1 = np.arange(n_nodes)
    idx2 = np.random.permutation(n_nodes)

    for i in range(n_nodes):
        G2.add_node(idx2[i], pos=G1.nodes[i]['pos'])
        for j in range(n_nodes):
            if i != j and (i, j) in G1.edges:
                G2.add_edge(idx2[i], idx2[j], eattr=G1.edges[i, j]['eattr'])

    # for i in range(n_nodes_r):
    #     if i > n_nodes:
    #         G2.add_node(i, pos=np.random.rand(3))
    #     else:
    #         G2.nodes[i]['pos'] = np.matmul(r.as_matrix(), G2.nodes[i]['pos'])
    return G1, G2, idx1, idx2

def network_plot_3D(fig, subplot, G, angle, save_folder=None):
    # set up the axes for the plot
    ax = fig.add_subplot(subplot, projection='3d')
    colors = [plt.cm.hsv(i / G.number_of_nodes()) for i in range(G.number_of_nodes())]

    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # 3D network plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for key, value in pos.items():
        xi = value[0]
        yi = value[1]
        zi = value[2]

        # Scatter plot
        ax.scatter(xi, yi, zi, c=colors[key], s=25, edgecolors='k', alpha=0.7)
        ax.text(xi, yi, zi + 0.01, str(key), color='red')

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted
    for i, j in enumerate(G.edges()):
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))

        # Plot the connecting lines
        ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)
    # ax.set_axis_off()


if __name__ == '__main__':
    sum_accuracy = 0
    loops = 1
    plot = True
    for i in range(loops):
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        G1, G2, idx1, idx2 = generate_rotation_3Dgraph(n_nodes=8, rotvect=np.array([0,0,np.pi/4]),
                                                       noise=0, connection_rate=0.1, seed=1)
        G1 = ARG(incoming_graph_data=G1)
        G2 = ARG(incoming_graph_data=G2)
        algorithm = GraphMatching(size=0, weight_range=0, connected_rate=0, noise_rate=0,
                                  ARG1=G1, ARG2=G2, idx1=idx1, idx2=idx2)
        match_matrix = algorithm.graph_matching()
        final_score, match1, match2 = GraphMatching.match_score(match_matrix, algorithm.idx1, algorithm.idx2)
        print(match_matrix)
        print('idx1, idx2', algorithm.idx1, algorithm.idx2)
        print('match1, match2', '\n', np.array(match1), '\n', np.array(match2))
        print(final_score)
        if plot:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            network_plot_3D(fig, 121, G1, angle=10, save_folder=None)
            network_plot_3D(fig, 122, G2, angle=10, save_folder=None)
            plt.show()
        print('----------')
        sum_accuracy += final_score
    print('accuracy = ', sum_accuracy / loops)