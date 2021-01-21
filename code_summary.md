Main file: `cyclic_flows.py`:
* Parse arguments
* Iterate through graph instances
    * Contract graph instance
    * Create SCC graph from contracted graph
    * Create an `Instance` object (defined in `flow.py`)
        * Attributes: graph (the contracted one), stuff for DP: topo order of
            graph, edge cuts, flow value, current `k` that we are testing,
            bounds on path weights in solution for this `k`, SCC graph and list
            of nodes in each SCC
