Main file: `cyclic_flows.py`:

Parse arguments

Iterate through graph instances
* Contract graph instance
* Create SCC graph from contracted graph
* Create an `Instance` object (defined in `flow.py`)
   * Attributes: graph (the contracted one), stuff for DP: topo order of
            graph, edge cuts, flow value, current `k` that we are testing,
            bounds on path weights in solution for this `k`, SCC graph and list
            of nodes in each SCC

* Get a lower bound on `k` (uses max cut size, and a second method)
* Starting from lower bound, start trying to solve using increasing `k`. This
    is all wrapped inside a function `find_opt_size` (still insice
    `cyclic_flows.py`), so that we can restrict this function to only run for
    `maxtime` time before moving on to next instance. Inside `find_opt_size`,
    we call a function called `solve` (defined in `guess_weights.py`)
    * takes in the `Instance` object (stores the current `k`) and the original
        graph (used to check that subpath constraints are satisfied, if we do
        find a solution)
    * tries guessing weights, so first sees if we can fix any weights for all
        solutions:
        * if `k`=size of largest cut, then the weights have to be the weights
            of the edges in that cut

