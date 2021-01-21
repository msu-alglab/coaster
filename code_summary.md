#### Main file: `cyclic_flows.py`

Parse arguments

Iterate through graph instances:
* Contract graph instance
* Create SCC graph from contracted graph
* Create an `Instance` object (defined in `flow.py`):
   * Attributes: graph (the contracted one), stuff for DP: topo order of
            graph, edge cuts, flow value, current `k` that we are testing,
            SCC graph and list of nodes in each SCC
    * Also holds upper and lower bounds for the `k` weights (in ascending
        order).
        * To find minimum: each top cut has a size. Call it `s`. We are trying
            to cover it with `k` paths. So, for each edge  with weight `w` in the cut, we can
            send at most `s - k + 1` paths over it, so we need a path of at
            least `w / (s - k + 1)`.
        * To find the maximum: note that the max weight for any path is minimum
            of max weights in any top cut.
* Get a lower bound on `k` (uses max cut size, and a second method)
* Starting from lower bound, start trying to solve using increasing `k`. This
    is all wrapped inside a function `find_opt_size` (still insice
    `cyclic_flows.py`), so that we can restrict this function to only run for
    `maxtime` time before moving on to next instance. Inside `find_opt_size`,
    we call a function called `solve` (defined in `guess_weights.py`):
    * Inputs: the `Instance` object (stores the current `k`) and the original
        graph (used to check that subpath constraints are satisfied, if we do
        find a solution)
    * Check if we can fix any weights for all
        solutions:
        * if `k`=size of largest cut, then the weights have to be the weights
            of the edges in that cut
        * if the range for the maximum weight is [x,x] for some x, we know the
            max weight
        * if any edge has weight 1, we know there is a path of weight 1
    * Fixes combinations of weights and tries to solve using dynamic
        programming by calling `solve` from `dp.py` (see below)

#### Dynamic programming: `dp.py`
