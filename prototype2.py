import numpy
from matplotlib import pylab, pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_nodes_from(range(1,10))
edge_list = [(1,2),(1,4),(2,5),(2,3),(3,6),(4,5),(4,7),(5,6),(5,8),(6,9),(7,8),(8,9)]
G.add_edges_from(edge_list)
# G.remove_node(4)
cg=nx.closeness_centrality(G)
dg=nx.degree_centrality(G)


H = nx.Graph()
H.add_nodes_from([1,2,3,5,6,7,8,9])
edge_list_H = [(1,3),(1,6),(1,9)]
H.add_edges_from(edge_list_H)
H.add_edges_from(edge_list)
ch=nx.closeness_centrality(H)
dh=nx.degree_centrality(H)

I = nx.Graph()
I.add_nodes_from([1,2,3,5,6,7,8])
edge_list_I = [(1,6),(1,8),(2,5),(2,3),(3,6),(5,6),(5,8),(7,8)]
I.add_edges_from(edge_list_I)
I.add_edges_from(edge_list)
ci=nx.closeness_centrality(I)
I.add_edges_from(edge_list_H)
di=nx.degree_centrality(I)


print("closeness of G: ",cg)
print("closeness of H: ",ch)
print("closeness of I: ",ci)
print("degre of G: ",dg)
print("degree of H: ",dh)
print("degree of I: ",di)

# nx.draw_networkx(H)
# pos = nx.spring_layout(H)
#
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.show()
# pylab.close()

