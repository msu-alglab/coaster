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
        * For the maximum weight:
            * To find minimum: each top cut has a size. Call it `s`. We are trying
                to cover it with `k` paths. So, for each edge  with weight `w` in the cut, we can
                send at most `s - k + 1` paths over it, so we need a path of at
                least `w / (s - k + 1)`.
            * To find the maximum: note that the max weight for any path is minimum
                of max weights in any top cut.
        * For all other weights:
            * To find upper bounds: look at edge weights in increasing order.
                Smallest is an upper bound. Then, for rest, if greater than sum
                of all upper bounds so far, must be an upper bound.
            * To find lower bounds: consider the weights in descending order.
                If the previous weights are equal to the upper bound, the
                following weights must account for all of the rest of the flow.
                So a lower bound is that remaining flow/# remaining paths.
* Get a lower bound on `k` (uses max cut size, and a second method)
* Starting from lower bound, start trying to solve using increasing `k`. This
    is all wrapped inside a function `find_opt_size` (still inside
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
    * Fixes combinations of weights found in graph and tries to solve using dynamic
        programming by calling `solve` from `dp.py` (see below). Combinations
        of weights get less restrictive, so the final call (if we made it
        there) would have no weights fixed.

#### Dynamic programming: `dp.py`

Primary function is `solve`:
* Takes in an `Instance` object, the original graph, and a length `k` list of guessed
    weights
* As in the Toboggan paper, the general idea is to generate routings (i.e.,
    sets of `k` paths) through the graph node by node in the topological
    ordering. Once all routings at vertex `v` are known, we can compute
    routings at vertex `v+1`. But each routing is actually stored as a `Constr`
    object, defined in `flow.py`:
    * Represent a `k` by `k + 1` matrix (the `k` by `k` matrix `A`
    and the solution vector `b`), which is always kept in reduced row echelon
    form (RREF). Each time a constraint is added, the matrix is
        converted to RREF.
    * After adding a new constraint and converting to RREF, if the linear system represented
     by a `Constr` object is of full rank, then
        it has a single solution. If this solution consists of positive integers
        (i.e., it is a valid flow), then the `Constr` object is replaced by a
        `SolvedConstr` object, which has the same interface as the `Constr` object
        but only stores the weights. If not, the `Constr` object is replaced by
        `None`.
* Additionally, the sets of paths themselves are stored in a `PathConf` object,
    which is also defined in `flow.py`:
    * `PathConf` objects match the path indices (all `k` of them) to vertices,
        and indicate which arc each path took to reach the vertex. A `PathConf`
        object can generate all the possible ways to extend itself into a new
        `PathConf` object by moving on to the next vertex in the topological
        ordering.
* First, create a `Constr` object with constraints from flow (all paths must
    sum to flow) and any guessed weights.
* Create a corresponding `PathConf` object representing the set of paths taken to the
    first node. (There is necessarily only one such set.)
* A dictionary maps `PathConf` pathsets to `Constr` constraint systems (or
    `SolvedConstr`, if solved)
* For each vertex in the topological ordering, for each `PathConf` and its
    corresponding `Constr` constraint system, use the `PathConf`
    `push` method to push the set of paths in the current `PathConf` to the
    next node. This generates (likely many) additional `PathConf` objects. For
    each of these, for each edge that is newly covered, add all of the
    corresponding constraints to the `Constr` object and add the (`PathConf`,
    `Constr`) pair to the table for this vertex.
* At the `t` vertex, there will be some set of (`PathConf`, `Constr`) pairs
    present. For each, if the `Constr` object is actually a `SolvedConstr`, then we know
    that we have found a solution (in the SCC graph). If it's still a `Constr`
    object, we will need to try all possible solutions. But this doesn't happen
    very often, and is addressed at the very end of this document.

At the end of the dynamic programming, we have a table full of `PathConf`
objects and their corresponding weights in `SolvedConstr` object.
For each one, we need to
recover all of the full paths through the graph that use these weights.
And for each path, we
also need to try to route it over the cycles and check if it satisfies the
subpath constraints. All of that is done in the `route_cycles_and_satisfy_subpath_constraints` method of a `SolvedConstr` object.

####`route_cycles_and_satisfy_subpath_constraints` method of `SolvedConstr` object (in `flow.py`):

####`route_cycles_and_satisfy_subpath_constraints` method of `Constr`
object (in `flow.py`): TODO
