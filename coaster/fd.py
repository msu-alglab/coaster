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
        self.create_reduced_graph()

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
        for sc, d in zip(self.graph.subpath_constraints,
                         self.graph.subpath_demands):
            self.reduced_graph.add_edge(sc[0], sc[-1], d)
            for u, v in zip(sc, sc[1:]):
                self.reduced_graph.remove_weight(d, u, v)
        self.reduced_graph.subpath_constraints = []
        self.reduced_graph.subpath_demands = []
        self.reduced_graph.check_flow()
        self.reduced_graph.write_graphviz("fd_reduced.dot")
