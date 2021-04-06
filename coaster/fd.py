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
        self.corresponding_edges = defaultdict(list)
        for sc, d in zip(self.graph.subpath_constraints,
                         self.graph.subpath_demands):
            self.reduced_graph.add_edge(sc[0], sc[-1], d)
            self.corresponding_edges[(sc[0], sc[-1])] = sc
            for u, v in zip(sc, sc[1:]):
                self.reduced_graph.remove_weight(d, u, v)
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
        """After a path solution has been found to the reduced graph, convert
        it back to a path solution in the original graph."""
        self.paths = []
        self.weights = self.reduced_graph.weights
        print("paths are", self.reduced_graph.paths)
        for path in self.reduced_graph.paths:
            nodeseq = self.reduced_graph.convert_arcseq_to_nodes(path)
            print("nodeseq is", nodeseq)
            rev_nodeseq = nodeseq[::-1]
            new_path = []
            # go through path backward to replace bridge edges with their
            # corresponding scs
            print("rev_nodeseq is", rev_nodeseq)
            for v, u in zip(rev_nodeseq, rev_nodeseq[1:]):
                print("v,u is", v, u)
                if (u, v) in self.corresponding_edges:
                    # two scs can start and end at the same node. how do we
                    # tell them apart?
                    print("is a sc")
                    new_path = self.corresponding_edges[(u, v)] + new_path
                else:
                    print("not an sc")
                    new_path = [v] + new_path
                print("new path is", new_path)
            self.paths.append(new_path)
            print(self.paths)
