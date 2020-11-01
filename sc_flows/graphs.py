#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
import copy
from collections import defaultdict
import itertools


class AdjList:
    def __init__(self, graph_file=None, graph_number=None, name=None,
                 num_nodes=None):
        self.graph_file = graph_file
        self.graph_number = graph_number
        self.name = name
        self.num_nodes_at_start = num_nodes
        self.vertices = set()
        self.adj_list = defaultdict(list)
        self.inverse_adj_list = defaultdict(list)
        self.out_arcs_lists = defaultdict(list)
        self.in_arcs_lists = defaultdict(list)
        self.arc_info = defaultdict(list)
        self.max_arc_label = 0
        self.subpath_constraints = list()
        self.subpath_demands = list()
        self.mapping = None

    def add_edge(self, u, v, flow):
        self.vertices.add(u)
        self.vertices.add(v)
        self.adj_list[u].append((v, flow))
        self.inverse_adj_list[v].append((u, flow))

        this_label = self.max_arc_label
        self.arc_info[this_label] = {
                                    'start': u,
                                    'destin': v,
                                    'weight': flow
        }
        self.out_arcs_lists[u].append(this_label)
        self.in_arcs_lists[v].append(this_label)
        self.max_arc_label += 1

    def add_subpath_constraint(self, L, d):
        self.subpath_constraints.append(L)
        self.subpath_demands.append(d)

    def print_subpath_constraints(self):
        print("Subpath constraints are:", self.subpath_constraints)
        print("Demands are:", self.subpath_demands)

    def out_arcs(self, node):
        return self.out_arcs_lists[node]

    def in_arcs(self, node):
        return self.in_arcs_lists[node]

    def arcs(self):
        return self.arc_info.items()

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def source(self):
        for v in self:
            if self.in_degree(v) == 0:
                return v
        raise TypeError("This graph has no source")

    def sink(self):
        for v in self:
            if self.out_degree(v) == 0:
                return v
        raise TypeError("This graph has no sink")

    def labeled_neighborhood(self, u):
        if u in self.adj_list:
            res = []
            for arc in self.out_arcs_lists[u]:
                dest = self.arc_info[arc]['destin']
                flow = self.arc_info[arc]['weight']
                res.append([dest, flow, arc])
            return res
        else:
            return []

    def neighborhood(self, u):
        if u in self.adj_list:
            return self.adj_list[u]
        else:
            return []

    def out_neighborhood(self, u):
        return self.neighborhood(u)

    def in_neighborhood(self, u):
        if u in self.inverse_adj_list:
            return self.inverse_adj_list[u]
        else:
            return []

    def remove_weight(self, weight, u, v):
        for (w, flow) in self.adj_list[u]:
            if w == v:
                self.adj_list[u].remove((w, flow))
                self.adj_list[u].append((w, flow - weight))
                self.inverse_adj_list[v].remove((u, flow))
                self.inverse_adj_list[v].append((u, flow - weight))
                break

    def copy(self):
        res = AdjList(self.graph_file, self.graph_number, self.name,
                      len(self))
        res.adj_list = copy.deepcopy(self.adj_list)
        res.inverse_adj_list = copy.deepcopy(self.inverse_adj_list)
        res.out_arcs_lists = copy.deepcopy(self.out_arcs_lists)
        res.in_arcs_lists = copy.deepcopy(self.in_arcs_lists)
        res.arc_info = copy.deepcopy(self.arc_info)
        res.subpath_constraints = copy.deepcopy(self.subpath_constraints)
        res.subpath_demands = copy.deepcopy(self.subpath_demands)
        res.vertices = set(self.vertices)
        return res

    def edges(self):
        for u in self.adj_list:
            for (v, flow) in self.adj_list[u]:
                yield (u, v, flow)

    def num_edges(self):
        return sum(1 for _ in self.edges())

    def reverse_edges(self):
        for v in self.inverse_adj_list:
            for (u, flow) in self.inverse_adj_list[v]:
                yield (u, v, flow)

    def out_degree(self, v):
        return len(self.out_neighborhood(v))

    def in_degree(self, v):
        return len(self.in_neighborhood(v))

    def fill_stack(self, v, visited, stack):
        """Add verts to a stack only after visiting all out-neighbors.
        For finding sccs."""
        visited[v] = True
        for u, _ in self.out_neighborhood(v):
            if not visited[u]:
                self.fill_stack(u, visited, stack)
        stack.append(v)

    def transpose(self):
        """Return a copy of the graph with all edge directions reversed."""
        res = self.copy()
        res.adj_list = res.inverse_adj_list
        return res

    def dfs(self, v, visited, this_scc):
        """traverse graph using DFS. For finding sccs."""
        visited[v] = True
        this_scc.append(v)
        for u, _ in self.out_neighborhood(v):
            if not visited[u]:
                self.dfs(u, visited, this_scc)

    def dfs_routing(self, v, visited, scc, end, this_route, routings):
        """For finding routings through a scc."""
        # print("beginning a call to dfs on node {}".format(v))
        visited[v] = True
        this_route.append(v)
        # print("visited is now", visited)
        if v == end:
            routings.append(this_route.copy())
            # print("at end. routings is now", routings)
        else:
            # print("Not at end. calling for out neighbors.")
            for out_neighb in\
                    (set([x[0] for x in self.out_neighborhood(v)]) & set(scc)):
                if not visited[out_neighb]:
                    self.dfs_routing(out_neighb, visited, scc, end,
                                     this_route, routings)
        visited[v] = False
        this_route.pop()

    def get_all_routings(self, start, end, scc):
        """
        Produce all routings from start node to end node through a scc
        in a cyclic graph.
        """
        visited = [False]*(max(scc) + 1)
        routings = []
        # print("visited is", visited)
        # print("Getting all routings from {} to {} through scc {}".format(
        #     start, end, scc))
        self.dfs_routing(start, visited, scc, end, [], routings)

        return routings

    def test_scc_flow_cover(scc, routing, weights):

        pass

    def route_cycle(self, scc, paths):
        """
        Try different routings through this scc using these paths until a
        viable one is found, or return None if there is no viable routing.
        """

        routings = dict()
        print("Paths to route over cycle (and in node, out node, and"
              " weight)")
        print(paths)
        unique_start_end_pairs = list(set([(x[1], x[2]) for x in
                                      paths]))
        pair_indices = dict()
        # only consider pairs that have different start/end
        for pair in unique_start_end_pairs:
            if pair not in routings:
                if pair[0] != pair[1]:
                    print("processing start/end", pair)
                    routings[pair] = self.get_all_routings(pair[0],
                                                           pair[1],
                                                           scc)
                    pair_indices[pair] = [i for i, x in
                                          enumerate(paths) if
                                          x[1] == pair[0] and x[2] ==
                                          pair[1]]

        # routings has all needed routings for this cycle.
        print("All routings are:", routings)
        print("All pair indices are:", pair_indices)
        weights = []
        for pair in pair_indices:
            weight_list = [x[3] for i, x in enumerate(paths)
                           if i in pair_indices[pair]]
            weights.append(weight_list)
        print("All weights are:", weights)
        # for each pair of start/end, create a product iterable over the
        # routings over the start/end repeated the number of times the
        # start/end pair occurs in this pathset
        products = [list(itertools.
                    product(routings[pair], repeat=len(pair_indices[pair])))
                    for pair in routings]
        for routing in itertools.product(*products):
            # check whether the routing is viable
            print("checking whether routing is viable:", routing)

            self.test_flow_cover(scc, routing, weights)

    def scc_contracted(self, sccs):
        """Return a copy of this graph contracted according to its subpath
        connected components.
        The first node in each scc is kept.
        """
        res = self.copy()
        for scc in sccs:
            if len(scc) > 1:
                v = scc[0]
                out_neighbs = [x[0] for x in res.out_neighborhood(v)]
                while len(set(out_neighbs) & set(scc)) > 0:
                    out_arcs = res.out_arcs_lists[v]
                    for arc in out_arcs:
                        if res.arc_info[arc]["destin"] in scc:
                            res.contract_edge(arc)
                    out_neighbs = [x[0] for x in res.out_neighborhood(v)]
        return res, sccs

    def scc(self):
        """
        Return a copy of the graph in which all strongly connected components
        are reduced to a single vertex, and a mapping from each vertex in the
        scc graph back to the original graph.
        """
        # find SCCs using Kosaraju's algorithm
        stack = []
        visited = [False]*(max(self.vertices) + 1)
        # start dfs at source
        self.fill_stack(self.source(), visited, stack)
        transpose = self.transpose()
        visited = [False]*(max(self.vertices) + 1)
        sccs = []
        while stack:
            v = stack.pop()
            this_scc = []
            if not visited[v]:
                transpose.dfs(v, visited, this_scc)
                sccs.append(this_scc)
        return self.scc_contracted(sccs)

    def contracted(self):
        """
        Return a copy of the graph in which all uv arcs where u has out degree
        1 or v has in degree 1 are contracted.
        """
        res = self.copy()
        # remembering which arcs were contracted in order to reconstruct the
        # paths in the original graph later
        arc_mapping = {e: [e] for e, _ in res.arcs()}
        # contract out degree 1 vertices
        for u in list(res):
            if res.out_degree(u) == 1:
                arc = res.out_arcs(u)[0]
                # mark u's inarcs to know they use the arc to be contracted
                for a in res.in_arcs(u):
                    arc_mapping[a].extend(arc_mapping[arc])
                # if u is the source, it has no in-arcs to mark the
                # contraction of this out-arc, so we store it in the out-arcs
                # of its out-neighbor.
                if res.in_degree(u) == 0:
                    v = res.out_neighborhood(u)[0][0]
                    for a in res.out_arcs(v):
                        new_path = list(arc_mapping[arc])
                        new_path.extend(arc_mapping[a])
                        arc_mapping[a] = new_path

                # contract the edge
                res.contract_edge(arc, keep_source=False)
        # contract in degree 1 vertices
        for v in list(res):
            if res.in_degree(v) == 1:
                arc = res.in_arcs(v)[0]
                # mark v's outarcs to know they use the arc to be contracted
                for a in res.out_arcs(v):
                    new_path = list(arc_mapping[arc])
                    new_path.extend(arc_mapping[a])
                    arc_mapping[a] = new_path
                # if u is the sink, it has no out-arcs to mark the contraction
                # of this in-arc, so we store it in the in-arcs of its
                # in-neighbor.
                if res.out_degree(v) == 0:
                    u = res.in_neighborhood(v)[0][0]
                    for a in res.in_arcs(u):
                        arc_mapping[a].extend(arc_mapping)

                # print("{} has in degree 1 from {}".format(v,u))
                res.contract_edge(arc, keep_source=True)
        res.mapping = arc_mapping
        return res, arc_mapping

    def contract_edge(self, e, keep_source=True):
        """
        Contract the arc e.

        If keep_source is true, the resulting vertex retains the label of the
        source, otherwise it keeps the sink's
        """
        u = self.arc_info[e]["start"]
        v = self.arc_info[e]["destin"]
        w = self.arc_info[e]["weight"]

        i = self.out_arcs(u).index(e)
        j = self.in_arcs(v).index(e)
        # move last neighbor into position of uv arc and delete arc
        self.adj_list[u][i] = self.adj_list[u][-1]
        self.adj_list[u] = self.adj_list[u][:-1]
        self.out_arcs_lists[u][i] = self.out_arcs_lists[u][-1]
        self.out_arcs_lists[u] = self.out_arcs_lists[u][:-1]

        # move last neighbor into position of uv arc and delete arc
        self.inverse_adj_list[v][j] = self.inverse_adj_list[v][-1]
        self.inverse_adj_list[v] = self.inverse_adj_list[v][:-1]
        self.in_arcs_lists[v][j] = self.in_arcs_lists[v][-1]
        self.in_arcs_lists[v] = self.in_arcs_lists[v][:-1]

        # to keep things concise, use the label a for the vertex to keep
        # and label b for the vertex to discard
        a, b = (u, v) if keep_source else (v, u)

        if a != b:
            # update out-neighbors of a
            self.adj_list[a].extend(self.out_neighborhood(b))
            self.out_arcs_lists[a].extend(self.out_arcs_lists[b])
            # make out-neighbors of b point back to a
            for lab, edge in zip(self.out_arcs(b), self.out_neighborhood(b)):
                w, f = edge
                i = self.inverse_adj_list[w].index((b, f))
                self.arc_info[lab]["start"] = a
                self.inverse_adj_list[w][i] = (a, f)

            # update in-neighbors of a
            self.inverse_adj_list[a].extend(self.in_neighborhood(b))
            self.in_arcs_lists[a].extend(self.in_arcs_lists[b])
            # make in neighbors of b point to a
            for lab, edge in zip(self.in_arcs(b), self.in_neighborhood(b)):
                w, f = edge
                i = self.adj_list[w].index((b, f))
                self.arc_info[lab]["destin"] = a
                self.adj_list[w][i] = (a, f)

            if b in self.adj_list:
                del self.adj_list[b]
            if b in self.inverse_adj_list:
                del self.inverse_adj_list[b]

            self.vertices.remove(b)
        del self.arc_info[e]

    def reversed(self):
        res = AdjList(self.graph_file, self.graph_number, self.name,
                      self.num_nodes())
        for u, v, w in self.edges():
            res.add_edge(v, u, w)
        return res

    def show(self, filename):
        import networkx as nx
        import matplotlib.pyplot as plt
        plt.clf()
        G = nx.DiGraph()
        for u, w, f in self.edges():
            G.add_edge(u, w)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.savefig(filename)

    def print_out(self):
        """Print the graph to screen."""
        for node in self.vertices:
            for arc in self.out_arcs_lists[node]:
                s = self.arc_info[arc]['start']
                t = self.arc_info[arc]['destin']
                w = self.arc_info[arc]['weight']
                print("{} {} {} arc_id={}".format(s, t, w, arc))


def test_paths(graph, pathset):
    for path in pathset:
        for i in range(len(path)-1):
            start = path[i]
            dest = path[i+1]
            for u, _ in graph.neighborhood(start):
                if u == dest:
                    break
            else:
                raise ValueError("Solution contains path with non-sequential"
                                 "vertices: {}, {}".format(start, dest))


def test_flow_cover(graph, solution):
    """Check that the path, weight solution covers every edge flow exactly and
    satisfies all subpath demands."""
    # Decode the solution set of paths
    recovered_arc_weights = defaultdict(int)
    for path_object in solution:
        path_deq, path_weight = path_object
        for arc in path_deq:
            recovered_arc_weights[arc] += path_weight
    # Check that every arc has its flow covered
    for arc, arc_val in graph.arc_info.items():
        true_flow = arc_val['weight']
        recovered_flow = recovered_arc_weights[arc]
        if (true_flow != recovered_flow):
            print("SOLUTION INCORRECT; arc {} has flow {},"
                  " soln {}".format(arc, true_flow, recovered_flow))


def convert_to_top_sorting(graph):
    # 1 temporary marked, 2 is finished
    source = graph.source()
    marks = {}

    def visit(n, ordering):
        if n in marks and marks[n] == 1:
            raise Exception('This graph is not a DAG: ' + graph.name)
        if n not in marks:
            marks[n] = 1
            for (m, _) in graph.neighborhood(n):
                visit(m, ordering)
            marks[n] = 2
            ordering.insert(0, n)

    ordering = []
    visit(source, ordering)

    return ordering


def compute_cuts(graph, ordering):
    """Compute the topological vertex cuts."""
    cuts = [None for v in graph]
    cuts[0] = set([graph.source()])

    for i, v in enumerate(ordering[:-1]):
        # Remove i from active set, add neighbors
        cuts[i+1] = set(cuts[i])
        cuts[i+1].remove(v)
        for t, w in graph.out_neighborhood(v):
            cuts[i+1].add(t)
    return cuts


def compute_edge_cuts(graph, ordering):
    """Compute the topological edge cuts."""
    # Contains endpoints and weights for arcs in each topological cut
    top_cuts = []
    # Contains the iteratively constructed top-edge-cut.
    # key is a node; value is a list of weights of arcs ending at key
    current_bin = defaultdict(list)

    # iterate over nodes in top ordering
    for v in ordering:
        v_neighborhood = graph.neighborhood(v)
        # remove from iterative cut-set the arcs ending at current node
        current_bin[v] = []
        for u, w in v_neighborhood:
            current_bin[u].append(w)
        # current cut-set done, add it to the output
        eC = []
        for u, weights in current_bin.items():
            eC.extend((u, weight) for weight in weights)
        top_cuts.append(eC)

    return top_cuts
