import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import random
from ARG import ARG
from graph_matching import GraphMatching


def generate_random_3Dgraph(n_nodes, radius, rotvect, noise, seed=None):
    if seed is not None:
        random.seed(seed)

    r = R.from_rotvec(rotvect)#[0, 0, np.pi / 2])
    # np.matmul(r.as_matrix(), np.array([1, 1, 1]))
    # Generate a dict of positions
    pos1 = {i: np.random.rand(3) for i in range(n_nodes)}
    G1 = nx.random_geometric_graph(n_nodes, radius, pos=pos1, p=10, seed=seed)
    for u, v in G1.edges:
        G1.edges[u, v]['eattr'] = 1

    G2 = nx.Graph()
    n_nodes_r = (1 + noise) * n_nodes

    idx1 = np.arange(n_nodes)
    idx2 = np.random.permutation(n_nodes)

    for i in range(n_nodes):
        print(idx2[i], G1.nodes[i]['pos'])
        G2.add_node(idx2[i], pos=G1.nodes[i]['pos'])
        for j in range(n_nodes):
            if i is not j and (i, j) in G1.edges:
                G2.add_edge(idx2[i], idx2[j], eattr=1)
                G2.add_edge(idx2[i], idx2[j], eattr=1)



    for i in range(n_nodes_r):
        if i > n_nodes:
            G2.add_node(i, pos=np.random.rand(3))
        else:
            G2.nodes[i]['pos'] = np.matmul(r.as_matrix(), G2.nodes[i]['pos'])
    return G1, G2, idx1, idx2

def network_plot_3D(fig, subplot, G, angle, save_folder=None):
    # set up the axes for the plot
    ax = fig.add_subplot(subplot, projection='3d')
    colors = [plt.cm.hsv(i / G.number_of_nodes()) for i in range(G.number_of_nodes())]

    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
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

    # r = R.from_rotvec([0, 0, np.pi / 2])
    # np.matmul(r.as_matrix(), np.array([1, 1, 1]))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.array([0, 1, 0])
    # y = np.array([0, 1, 1])
    # z = np.array([0, 1, 0])
    # ax.plot(x, y, z)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.show()

    G1, G2, idx1, idx2 = generate_random_3Dgraph(n_nodes=10, radius=0.25, rotvect=np.array([0,0,np.pi/4]),
                                     noise=0, seed=1)
    G1 = ARG(incoming_graph_data=G1)
    G2 = ARG(incoming_graph_data=G2)
    algorithm = GraphMatching(size=0, weight_range=0, connected_rate=0, noise_rate=0,
                              ARG1=G1, ARG2=G2, idx1=idx1, idx2=idx2)
    match_matrix = algorithm.graph_matching()
    final_score = GraphMatching.match_score(match_matrix, algorithm.idx1, algorithm.idx2)
    print(final_score)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    network_plot_3D(fig, 121, G1, 10, save_folder=None)
    network_plot_3D(fig, 122, G2, 10, save_folder=None)
    plt.show()