#%%
%load_ext autoreload
%autoreload 2

import copy
import math
from typing import Any, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq
import scipy.spatial
from matplotlib import collections as mc


#%%
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    retval = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return retval


def create_sub_system(center):
    points = np.random.randn(100, 2)
    # Kill extreme points
    points = points[np.amax(points, axis=1) <= 3]
    points = points[np.amin(points, axis=1) >= -3]

    centroids, _ = scipy.cluster.vq.kmeans(
        points, 6, iter=100, thresh=1e-05, check_finite=True
    )
    centroids += center
    return centroids

def create_spatial_graph():
    points = np.vstack(
        [create_sub_system([x, y]) for x in range(0, 16, 4) for y in range(0, 16, 4)]
    )

    tri = scipy.spatial.Delaunay(points)

    # create a set for edges that are indexes of the points
    edges: Set[Tuple[int, int]] = set()
    for n in range(tri.nsimplex):
        # Skip triangles that are too thin
        skip = False
        for x in range(3):
            if (
                angle_between(
                    points[tri.vertices[n, x]] - points[tri.vertices[n, (x + 1) % 3]],
                    points[tri.vertices[n, (x + 2) % 3]]
                    - points[tri.vertices[n, (x + 1) % 3]],
                )
                < math.pi / 10
            ):
                skip = True
        if skip:
            continue
        edge = sorted([tri.vertices[n, 0], tri.vertices[n, 1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([tri.vertices[n, 0], tri.vertices[n, 2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([tri.vertices[n, 1], tri.vertices[n, 2]])
        edges.add((edge[0], edge[1]))

    trimmed_edges = []
    for edge in list(edges):
        line = (points[edge[0]], points[edge[1]])
        dist = scipy.spatial.distance.euclidean(line[0], line[1])
        if dist < 2.0:
            trimmed_edges.append(edge)

    min_point = list(points[0])
    for point in points:
        min_point[0] = min(point[0], min_point[0])
        min_point[1] = min(point[1], min_point[1])

    scaled_int_points = []
    for point in points:
        scaled_int_points.append(
            (
                int((point[0] - min_point[0] + 1.0) * 100),
                int((point[1] - min_point[1] + 1.0) * 100),
            )
        )
    return (scaled_int_points, trimmed_edges)


#%%
import networkx as nx


def draw_graph(g:nx.Graph):
    pass    

#%%
np.random.seed(2)


points, edges = create_spatial_graph()
points_dict = dict([(idx, p) for idx,p in enumerate(points)])
np_points = np.vstack(points)


# %%

map_graph=nx.Graph()
map_graph.add_edges_from(edges)

draw_params = {
    "node_size":100,
    "font_size":20,
    "node_color":"r",
    "verticalalignment":'top',
}

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(map_graph, pos=np_points, ax=ax, **draw_params)

# %%
spanning_tree=nx.minimum_spanning_tree(map_graph)

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(spanning_tree, pos=np_points, ax=ax, **draw_params)

# %%
chain_edges=list(nx.chain_decomposition(map_graph, root=0))
chain_graph = nx.Graph()
chain_graph.add_edges_from(chain_edges[0])

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(chain_graph, pos=np_points, ax=ax, **draw_params)


# %%
bridge_edges=list(nx.bridges(map_graph, root=0))
bridge_graph = nx.Graph()
bridge_graph.add_edges_from(bridge_edges)

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(bridge_graph, pos=np_points, ax=ax, **draw_params)

# %%
node_centrality=nx.closeness_centrality(map_graph)
node_centrality = dict([(k,round(v,2)) for k,v in node_centrality.items()])

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(map_graph, pos=np_points, labels=node_centrality, ax=ax, **draw_params)

# %%
graph_with_flow = copy.deepcopy(map_graph)
nx.set_edge_attributes(graph_with_flow, 1.0, "capacity")
max_flow, max_flow_per_edge=nx.maximum_flow(graph_with_flow, 0, 73)
print(max_flow)
print(max_flow_per_edge)

edge_labels = {}
for source, dest_flow_pairs in max_flow_per_edge.items():
    for destination, flow in dest_flow_pairs.items():
        if flow > 0:
            edge_labels[(source, destination)] = int(flow)

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(map_graph, pos=np_points, with_labels=False, ax=ax, **draw_params)
nx.draw_networkx_edge_labels(map_graph, pos=np_points, edge_labels=edge_labels)


# %%
graph_with_flow = copy.deepcopy(map_graph)
nx.set_edge_attributes(graph_with_flow, 1.0, "capacity")
max_flow, max_flow_per_edge=nx.maximum_flow(graph_with_flow, 1, 74)

edge_labels = {}
for source, dest_flow_pairs in max_flow_per_edge.items():
    for destination, flow in dest_flow_pairs.items():
        if flow > 0:
            edge_labels[(source, destination)] = int(flow)

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.set_xlim((0,2000))
ax.set_ylim((0,2000))
nx.draw_networkx(map_graph, pos=np_points, with_labels=False, ax=ax, **draw_params)
nx.draw_networkx_edge_labels(map_graph, pos=np_points, edge_labels=edge_labels)

# %%
