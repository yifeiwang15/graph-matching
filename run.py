from datetime import datetime
import numpy as np
import pandas as pd

from twin_graphs_generator import generate_twin_graphs
from ARG import ARG
from graph_matching import GraphMatching
import networkx as nx
import matplotlib.pyplot as plt

def run_graph_matching(size=4, weight_range=1, connected_rate=0.5, noise_rate=0.2):

    algorithm = GraphMatching(size, weight_range, connected_rate, noise_rate)
    match_matrix = algorithm.graph_matching()
    final_score = GraphMatching.match_score(match_matrix, algorithm.idx1, algorithm.idx2)
    return final_score, algorithm

def draw_graph(g, subplot):
    plt.subplot(subplot)
    nx.draw_networkx(g)

if __name__ == '__main__':
    print("start")
    score, algorithm = run_graph_matching()
    print(score)
    draw_graph(algorithm.ARG1, 121)
    draw_graph(algorithm.ARG2, 122)
    plt.show()




# try different parameter combinations of size, connected_rate and noise_rate.3
# Further add the stochastic property.
# (But I am not sure its intuition, should further discussion!)

# For every combination, we repeat 20 times.
# Record mean accuracy and std.
# Should also record the runnning time (mean and std).

# The result should be saved as a dataframe.

#
# size = 20
# repeat = 1
# connected_rate_list = [0.2, 0.3, 0.5]
# noise_rate_list = [0, 0.1, 0.2, 0.3, 0.5]
# acc_mean = np.zeros([len(connected_rate_list), len(noise_rate_list)])
# acc_std = np.zeros([len(connected_rate_list), len(noise_rate_list)])
# time_mean = np.zeros([len(connected_rate_list), len(noise_rate_list)])
# time_std = np.zeros([len(connected_rate_list), len(noise_rate_list)])
# for i in range(len(connected_rate_list)):
#     for j in range(len(noise_rate_list)):
#         acc = np.zeros(repeat)
#         time = np.zeros(repeat)
#         for rep in range(repeat):
#             start = datetime.now()
#             acc[rep] = run_graph_matching(size=size, weight_range=1, connected_rate=connected_rate_list[i], noise_rate=noise_rate_list[j])
#             end = datetime.now()
#             time[rep] = (end-start).seconds
#         print(str(end) + ' | ' + 'c_rate: '+ str(connected_rate_list[i]) + ' | ' + 'n_rate: '+ str(noise_rate_list[j]))
#         print(str(np.mean(acc))+' | ' + str(np.std(acc)) + ' | ' + str(np.mean(time))+' | ' + str(np.std(time)))
#         acc_mean[i,j] = np.mean(acc)
#         acc_std[i,j] = np.std(acc)
#         time_mean[i,j] = np.mean(time)
#         time_std[i,j] = np.std(time)
# acc_mean = pd.DataFrame(acc_mean,columns=noise_rate_list, index=connected_rate_list)
# acc_std = pd.DataFrame(acc_std,columns=noise_rate_list, index=connected_rate_list)
# time_mean = pd.DataFrame(time_mean,columns=noise_rate_list, index=connected_rate_list)
# time_std = pd.DataFrame(time_std,columns=noise_rate_list, index=connected_rate_list)
#
# acc_mean.to_csv("acc_mean_m2.csv")
# acc_std.to_csv("acc_std_m2.csv")
# time_mean.to_csv("time_mean_m2.csv")
# time_std.to_csv("time_std_m2.csv")
