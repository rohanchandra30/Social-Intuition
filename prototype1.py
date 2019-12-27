import numpy
import networkx as nx
import pydot
from networkx import waxman_graph
from matplotlib import pylab, pyplot as plt

# G = waxman_graph(10)
# nx.draw(G)
# print(G)


def save_graph(graph,file_name):
#initialze Figure
    nx.draw_networkx(graph)
    # pos = nx.spring_layout(graph)
    # nx.draw_networkx_nodes(graph,pos)
    # nx.draw_networkx_edges(graph,pos)
    # nx.draw_networkx_labels(graph,pos)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()

#Assuming that the graph g has nodes and edges entered
G = waxman_graph(10,domain=(0, 0, .1, .1))
# G = nx.barbell_graph(3,5)
save_graph(G,"my_graph.jpg")

#it can also be saved in .svg, .png. or .ps formats