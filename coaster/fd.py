from collections import defaultdict


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
        print("demands is", demands)
        for arc_id in self.graph.adj_list:
            u = self.graph.arc_info[arc_id]["start"]
            v = self.graph.arc_info[arc_id]["destin"]
            f = self.graph.arc_info[arc_id]["weight"]
            if demands[(u, v)] > f:
                self.overdemands[(u, v)] = demands[(u, v)] - f

    def create_reduced_graph(self):
        """
        Convert each subpath constraint into a bridge edge, and subtract its
        demand from the flow on the edges it covers
        """
        print(self.reduced_graph.adj_list)
        self.corresponding_edges = defaultdict(list)
        for sc_nodes, d in zip(self.graph.subpath_constraints,
                               self.graph.subpath_demands):
            sc = self.reduced_graph.convert_nodeseq_to_arcs(sc_nodes)
            print("reducing based on sc (nodes)", sc_nodes)
            print("arc id based:", sc)
            start = self.reduced_graph.arc_info[sc[0]]["start"]
            end = self.reduced_graph.arc_info[sc[-1]]["destin"]
            new_arc_id = self.reduced_graph.add_edge(start, end, d)
            self.corresponding_edges[new_arc_id] = sc_nodes
            for arc_id in sc:
                self.reduced_graph.remove_weight_by_arc_id(d, arc_id)
        self.reduced_graph.subpath_constraints = []
        self.reduced_graph.subpath_demands = []
        self.reduced_graph.check_flow()
        self.reduced_graph.write_graphviz("fd_reduced.dot")

    def solve(self):
        """
        Step 1: get rid of overdemanded edges (not implemented).
        Step 2: create FD reduction.
        Step 3: find a path solution.
        """
        self.create_reduced_graph()
        self.reduced_graph.run_greedy_width()

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
        # combine paths if same
        path_to_weight = defaultdict(int)
        for path, weight in zip(self.paths, self.weights):
            path_to_weight[tuple(path)] += weight
        self.paths = [list(x) for x in list(path_to_weight.keys())]
        self.weights = list(path_to_weight.values())

