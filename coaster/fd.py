from collections import defaultdict
from itertools import combinations
from coaster.graphs import test_flow_cover


class ExactFlowInstance:
    """
    A class for executing the flow decomposition based heuristic for the
    FDSC problem.
    """
    def __init__(self, graph):
        self.graph = graph
        self.overdemands = defaultdict(int)
        self.compute_overdemands()
        # putting this in for now since I'm just going to focus on instances
        # without overdemanded edges
        if len(self.overdemands) > 0:
            raise Exception("Overdemanded edge", self.overdemands)
        self.reduced_graph = self.graph.copy()

    def compute_overdemands(self):
        demands = defaultdict(int)
        for sc, d in zip(self.graph.subpath_constraints,
                         self.graph.subpath_demands):
            for u, v in zip(sc, sc[1:]):
                demands[(u, v)] += d
        for arc_id in self.graph.adj_list:
            u = self.graph.arc_info[arc_id]["start"]
            v = self.graph.arc_info[arc_id]["destin"]
            f = self.graph.arc_info[arc_id]["weight"]
            if demands[(u, v)] > f:
                self.overdemands[(u, v)] = demands[(u, v)] - f

    def get_arc_from_sc(self, sc):
        for (key, value) in self.corresponding_edges.items():
            if value == sc:
                return key

    def create_reduced_graph(self):
        """
        Convert each subpath constraint into a bridge edge, and subtract its
        demand from the flow on the edges it covers
        """
        self.corresponding_edges = defaultdict(list)
        for sc_nodes, d in zip(self.graph.subpath_constraints,
                               self.graph.subpath_demands):
            sc = self.reduced_graph.convert_nodeseq_to_arcs(sc_nodes)
            start = self.reduced_graph.arc_info[sc[0]]["start"]
            end = self.reduced_graph.arc_info[sc[-1]]["destin"]
            new_arc_id = self.reduced_graph.add_edge(start, end, d)
            self.corresponding_edges[new_arc_id] = sc_nodes
            for arc_id in sc:
                self.reduced_graph.remove_weight_by_arc_id(d, arc_id)
        self.reduced_graph.subpath_constraints = []
        self.reduced_graph.subpath_demands = []
        self.reduced_graph.check_flow()

    def solve(self, no_br=False):
        """
        Step 1: get rid of overdemanded edges (not implemented).
        Step 2: create FD reduction.
        Step 3: find a path solution.
        """
        self.create_reduced_graph()
        if not no_br:
            self.increase_bridge_flows()
        self.reduced_graph.run_greedy_width()

    def increase_bridge_flows(self):
        """
        In order to create better path decompositions, try to route as much
        flow as possible on bridge edges.
        """
        # print("\nIncreasing bridge flows.")
        for sc in self.graph.subpath_constraints:
            # print("sc is", sc)
            arcs = self.graph.convert_nodeseq_to_arcs(sc)
            # see if we can increase the bridge edge
            min_f = self.reduced_graph.arc_info[arcs[0]]["weight"]
            # print("min_f for this sc is", min_f)
            for arc in arcs:
                min_f = min(self.reduced_graph.arc_info[arc]["weight"], min_f)
                # print("arc from", self.reduced_graph.arc_info[arc]["start"],
                #       "to",
                #       self.reduced_graph.arc_info[arc]["destin"], "has  weight",  # noqa
                #       self.reduced_graph.arc_info[arc]["weight"])
            # change bridge edge flow and arc flows
            sc_arc = self.get_arc_from_sc(sc)
            self.reduced_graph.remove_weight_by_arc_id(-min_f, sc_arc)
            for arc in arcs:
                self.reduced_graph.remove_weight_by_arc_id(min_f, arc)
            self.reduced_graph.check_flow()
        self.reduced_graph.write_graphviz("fd_reduced.dot")

    def convert_paths(self):
        """
        After a path solution has been found to the reduced graph, convert
        it back to a path solution in the original graph.
        If there are mutiple copies of the same path (follow the same sequence
        of nodes), combine into one.
        """
        self.paths = []
        self.weights = self.reduced_graph.weights
        for path in self.reduced_graph.paths:
            new_path = []
            for arc_id in path:
                if arc_id in self.corresponding_edges:
                    new_path += self.corresponding_edges[arc_id]
                else:
                    # if this is the first arc_id we are adding, add both first
                    # and last node
                    if len(new_path) == 0:
                        new_path +=\
                            [self.reduced_graph.arc_info[arc_id]["start"]]
                    new_path += [self.reduced_graph.arc_info[arc_id]["destin"]]

            self.paths.append(new_path)
        self.dedupe_paths()

    def dedupe_paths(self):
        """Combine paths if same."""
        path_to_weight = defaultdict(int)
        for path, weight in zip(self.paths, self.weights):
            path_to_weight[tuple(path)] += weight
        self.paths = [list(x) for x in list(path_to_weight.keys())]
        self.weights = list(path_to_weight.values())

    def splice_heuristic(self):
        """
        After we have converted paths back into the orignal graph, look for
        opportunities to splice pairs of weight 1 paths so that they match a
        non weight 1 path.
        """
        k = len(self.paths)
        print("initial length of paths is", k)
        print("Initial paths are:")
        for x in self.paths:
            print(x)
        spliced = True
        while spliced:
            spliced = False
            for pair in combinations([i for i, x in enumerate(self.paths)], 2):
                if self.weights[pair[0]] == self.weights[pair[1]]:
                    a = self.paths[pair[0]]
                    b = self.paths[pair[1]]
                    print("a=", a)
                    print("b=", b)
                    for c in [x for i, x in enumerate(self.paths) if i not in
                              pair]:
                        # if a and b can be spliced to make c, do it
                        spliced = self.splice(a, b, c, pair)
                        if spliced:
                            break
                    if spliced:
                        self.dedupe_paths()
                        print("spliced")
                        print("len of paths is", len(self.paths))
                        break
        print("Reduced number of paths by", k - len(self.paths))

    def splice(self, a, b, c, pair):
        """Check whether a and b can be spliced to make c, and if so, do it."""
        spliced = False
        prefix = lcp(a, c)
        suffix = lcs(c, b)
        if len(prefix) + len(suffix) >= len(c) + 1:
            # make the start of the new first path and suffix the end
            num_nodes_from_b = len(c) - len(prefix)
            a_prime = prefix + b[-num_nodes_from_b:]
            b_prime = b[:len(b) - num_nodes_from_b] +\
                a[len(prefix):]
            print("a':", a_prime)
            print("b':", b_prime)
            self.paths[pair[0]] = a_prime
            self.paths[pair[1]] = b_prime
            try:
                test_flow_cover(self.graph, self.paths, self.weights)
            except AssertionError:
                self.paths[pair[0]] = a
                self.paths[pair[1]] = b
            else:
                print("reassigned paths. new paths are:")
                print(self.paths[pair[0]])
                print(self.paths[pair[1]])
                assert self.paths[pair[0]] == c
                spliced = True
        return spliced


def lcp(s1, s2):
    index = 0
    ch1 = s1[index]
    ch2 = s2[index]
    while ch1 == ch2 and index < len(s1) and index <= len(s2):
        index += 1
        ch1 = s1[index]
        ch2 = s2[index]
    return s1[:index]


def lcs(s1, s2):
    s1 = list(reversed(s1))
    s2 = list(reversed(s2))
    rev_prefix = lcp(s1, s2)
    rev_prefix.reverse()
    return rev_prefix
