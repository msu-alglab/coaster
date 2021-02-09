#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# python libs
import math
from collections import defaultdict, deque
import itertools
import numpy as np
from scipy.optimize import linprog
import copy
import time

# local imports
from coaster.graphs import convert_to_top_sorting, compute_cuts,\
                            compute_edge_cuts
from coaster.partition import algorithm_u


class Instance:
    """
    Information about an input instance to flow decomposition.

    Maintains a topological ordering of the graph, the flow, and bounds on the
    feasible path weights.
    """

    def __init__(self, graph, k=None, cyclic_graph=None,
                 sccs=None):
        """Create an instance from a graph and guess for the solution size."""
        # information about the graph and its ordering
        self.graph = graph
        self.ordering = convert_to_top_sorting(graph)
        self.cuts = compute_cuts(self.graph, self.ordering)
        self.edge_cuts = compute_edge_cuts(self.graph, self.ordering)
        self.flow = sum(w for _, w in self.graph.neighborhood(
                                            self.graph.source()))
        # get a lower bound on the number of paths needed
        # We pass the input k to this function so it can inform the user if
        # the input k can be easily identified as too small.
        self.k = self._optimal_size_lower_bound(k)

        # our initial guesses for weights will come from the flow values
        self.weights = sorted(set([w for _, _, w in self.graph.edges()]))

        # compute bounds on the largest weight
        self.max_weight_bounds = self._compute_max_weight_bounds()
        # compute bounds on the individual weights
        self.weight_bounds = self._compute_weight_bounds()

        # cyclic_graph
        self.cyclic_graph = cyclic_graph
        # sccs in the original cyclic graph
        self.sccs = sccs

    def info(self):
        """A string representation of this object."""
        print("n = {}, m = {}, k = {}.".format(len(self.graph),
                                               self.graph.num_edges(), self.k))
        print("Weights:", self.weights)
        print("Max-weight bounds:", self.max_weight_bounds)
        print("Weight bounds:", list(map(tuple, self.weight_bounds)))
        # print("")
        # print("Cut-representation:")
        # print(self.cuts)

    def _compute_max_weight_bounds(self):
        # Get lower bound for highest weight
        min_max_weight = 1
        # This checks each topological cut
        for topol_cut in self.edge_cuts:
            cut_size = len(topol_cut)
            for _, w in topol_cut:
                # use pigeonhole principle to lowerbound max weight
                min_max_weight = max(min_max_weight, w // (self.k-cut_size+1))

        # Compute heaviest possible path in graph
        # by iterating over each node's out-neighborhood
        maxpath = {v: 0 for v in self.graph}
        maxpath[self.graph.source()] = self.flow
        for v in self.ordering:
            out = self.graph.neighborhood(v)
            for u, w in out:
                maxpath[u] = max(maxpath[u], min(w, maxpath[v]))

        return (min_max_weight, maxpath[self.graph.sink()])

    def _compute_weight_bounds(self):
        supseq = []
        summed = 0
        # supseq is a list of "super-increasing" values taken from edge weights
        # starting from smallest weight in the graph. These values are upper
        # bounds on the different path weights.
        for w in self.weights:
            if w > self.max_weight_bounds[1]:
                break
            if w > summed:
                supseq.append(w)
                summed += w
        # pad the rest of supseq with the max_weight_bound
        while len(supseq) < self.k:  # Sentinel elements
            supseq.append(self.max_weight_bounds[1])

        bounds = [(1, w) for w in supseq[:self.k]]
        bounds[-1] = self.max_weight_bounds

        # Next, compute lowerbounds for the path weights.
        uppersum = [u for _, u in bounds]
        for i in reversed(range(self.k-1)):
            uppersum[i] += uppersum[i+1]

        # Refine lower bounds by using upper bounds:
        # the weight of path i must be at least F_i / i
        # where F_i is an upper bound on how much flow all paths > i
        # take up.
        for i in range(1, self.k-1):
            lower = max(bounds[i][0], (self.flow-uppersum[i+1]) // (i+1))
            bounds[i] = (lower, bounds[i][1])
        return np.array(bounds)

    def _compute_multiset_bound(self, list1, list2):
        """
        Treat twolists as multisets, return list1-list2.
        Note: input lists should contain int or float type.
        """
        # convert to dicts with contents as keys, multiplicities as vals
        size1 = len(list1)
        size2 = len(list2)

        dict1 = defaultdict(int)
        for item in list1:
            dict1[item] += 1
        dict2 = defaultdict(int)
        for item in list2:
            dict2[item] += 1
        num_repeated = 0
        for key, val in dict1.items():
            num_repeated += min(val, dict2[key])
        size1 -= num_repeated
        size2 -= num_repeated
        return num_repeated + math.ceil(max(size1, size2)/3) + \
            min(size1, size2)

    def _optimal_size_lower_bound(self, k):
        """
        Get a lower bound on the optimal solution size.

        We look over all s-t edge cuts consistent with the topological ordering
        and pick the largest. Then we look over all pairs of cut-sets that are
        large enough to further improve this lower-bound and check whether the
        number of distinct edge-weights requires a larger lower-bound than
        merely the largest cut-set size.
        """
        edge_cut_sizes = [len(C) for C in self.edge_cuts]
        max_edge_cut = max(edge_cut_sizes)
        lower_bound = max_edge_cut
        self.max_edge_cut_size = max_edge_cut

        # Now check all pairs of cutsets "large enough" for better bound
        sorted_cut_sizes = sorted([(cut_size, which_cut) for which_cut,
                                   cut_size in enumerate(edge_cut_sizes)],
                                  reverse=True)
        cutsets_of_best_bound = []
        # Starting with largest, iterate over cutsets
        for idx1 in range(len(sorted_cut_sizes)):
            current_size1, which_cut1 = sorted_cut_sizes[idx1]
            # once one set is too small, all following will be, so break out
            if math.ceil(current_size1/3) + current_size1 <= lower_bound:
                break
            for idx2 in range(idx1+1, len(sorted_cut_sizes)):
                current_size2, which_cut2 = sorted_cut_sizes[idx2]
                # if cutsize2 too small, the rest will be: break inner for loop
                temp_bound = min(current_size1, current_size2) + math.ceil(
                    max(current_size1, current_size2)/3)
                if temp_bound <= lower_bound:
                    break
                # Now compute actual bound for this pair of cutsets;
                # Get weights for each cutset as a multiset,
                # compute size of (larger) difference
                weights1 = set([w for _, w in self.edge_cuts[which_cut1]])
                weights2 = set([w for _, w in self.edge_cuts[which_cut2]])
                bound = self._compute_multiset_bound(weights1, weights2)
                # Check if we need to update bound
                if bound > lower_bound:
                    lower_bound = bound
                    cutsets_of_best_bound = [which_cut1, which_cut2]
        if len(cutsets_of_best_bound) > 0:
            which_cut1, which_cut2 = cutsets_of_best_bound

        # let the user know their guess was bad if it was
        self.best_cut_lower_bound = lower_bound
        print("# Preprocessing")
        print("#\tGraph has an edge cut of size {}.\n"
              "#\tInvestigating cutsets yields bound {}.\n"
              "#\tUser supplied k value of {}.\n"
              "#\tContinuing using k = {}"
              "".format(max_edge_cut, lower_bound, k, lower_bound))
        if k is not None and lower_bound > k:
            return lower_bound
        elif k is None:
            return lower_bound
        else:
            return k

    def try_larger_k(self):
        """
        Increase the value of k by 1.

        We need to do this in a method in order to update internal data
        structures about the weights.
        """
        self.k = self.k + 1
        # compute bounds on the largest weight
        self.max_weight_bounds = self._compute_max_weight_bounds()
        # compute bounds on the individual weights
        self.weight_bounds = self._compute_weight_bounds()

    def has_bad_bounds(self):
        """Check whether weight bounds disallow all solutions."""
        # upper weight bounds miss each other
        if self.max_weight_bounds[0] > self.max_weight_bounds[1]:
            return True
        # lower and upper bounds of each weight position miss each other
        for lower, upper in self.weight_bounds:
            if lower > upper:
                return True
        # otherwise all good
        return False


class Constr:
    """
        Class representing linear constraints imposed on path
        weights as collected by the DP routine.
    """
    eps = np.finfo(float).eps
    ORDER_MATRIX = {}  # Pre-computed matrices with ones on the diagonal and
    #                    -1 on the upper off-diagonal.
    ZERO_VECS = {}     # Pre-computed zero vectors
    INFEASIBLE = 0
    REDUNDANT = 1
    VALID = 2
    SOLVED = 3
    POW2 = None

    def __init__(self, instance=None, constraint=None):
        if constraint is not None:
            self.instance = constraint.instance
            self.known_values = copy.copy(constraint.known_values)
            self.hashvalue = copy.copy(constraint.hashvalue)
            self.rank = copy.copy(constraint.rank)
            self.utri = constraint.utri.copy()
            self.pivot_lookup = copy.copy(constraint.pivot_lookup)
        else:
            self.instance = instance
            self.known_values = [None] * self.instance.k
            # Make sure the necessary constants exist
            if self.instance.k not in Constr.ORDER_MATRIX:
                t = self.instance.k
                Constr.ORDER_MATRIX[t] = np.eye(t-1, t, dtype=int) - \
                    np.eye(t-1, t, k=1, dtype=int)
                Constr.ZERO_VECS[t-1] = np.zeros(t-1, dtype=int)
                Constr.POW2 = 2**np.arange(64, dtype=np.uint64)

            row = np.array([1] * self.instance.k + [self.instance.flow])
            # In our application instance.k and instance.flow should always be
            # the same, but we want to keep things clean.
            self.hashvalue = hash(row.data.tobytes()) ^\
                hash(self.instance.k) ^\
                hash(self.instance.flow)
            self.rank = 1
            self.utri = np.zeros((self.instance.k, self.instance.k+1))
            self.utri[0] = row
            # pivot_lookup[j] gives the row_index of pivot in column j
            # This is to avoid having to permute utri to be in RREF
            self.pivot_lookup = [-1 for j in range(len(row))]
            self.pivot_lookup[0] = 0

    def __repr__(self):
        return str(self.utri)

    def _copy_with_new_row(self, row, reduced_row, pivot_idx):
        res = Constr(constraint=self)

        # update hashvalue by new row
        res.hashvalue ^= hash(row.data.tobytes())
        res.rank = self.rank + 1

        # Ensure res.utri (with row added) is in RREF form.
        # Make sure pivot is a 1
        pivot_value = reduced_row[pivot_idx]
        if pivot_value != 1:
            reduced_row = reduced_row/pivot_value
        # use new pivot to eliminate in other rows
        for idx in range(self.rank):
            val = res.utri[idx, pivot_idx]
            if val != 0:
                res.utri[idx, :] = res.utri[idx, :] - reduced_row*val
        # update the resulting utri
        res.utri[self.rank, :] = reduced_row
        res.pivot_lookup[pivot_idx] = self.rank

        return res

    def is_redundant(self):
        # We can reduce the number of redundant solutions by imposing that
        # the already known path-weights are sorted.
        weights = [w for w in self.known_values if w is not None]
        if not all(weights[i] <= weights[i+1] for i in range(len(weights)-1)):
            return True

        # The following LP tells us whether there is a feasible (rational)
        # solution whose weights are sorted in ascending order. If that is
        # not the case this set of constraints is redundant.
        t = self.instance.k
        c = np.array([1]*t)  # Optimization not important.
        # Equality constraints: flow values
        A_eq = self.utri[:self.rank, :-1]
        b_eq = self.utri[:self.rank, -1]
        # Inequality constraints: ensure that weights are sorted
        A_ub = Constr.ORDER_MATRIX[t]
        assert(A_ub.shape == (t-1, t))
        b_ub = Constr.ZERO_VECS[t-1]
        # Bounds for flow values: at least 1, at most F-(t-1) (since non-zero)
        bounds = self.instance.weight_bounds

        # wrap linear program in try block to check when we have bounds that
        # are invalid.
        try:
            optres = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
            if optres.status == 2:
                # System is infeasible and therefore redundant.
                return True
        except ValueError as e: # noqa
            # bounds that don't make sense indicate an infeasible system
            return True

        return False

    def _check_lin_dep(self, input_vector):
        """Check if input_vector is in rowspan of A."""
        current_pivot = len(input_vector) + 1
        vector = input_vector.copy()
        for col_idx in range(len(vector)):
            val = vector[col_idx]
            if val != 0:
                # check for pivot above it
                row_idx = self.pivot_lookup[col_idx]
                if row_idx != -1:  # if vector entry lies under a pivot, reduce
                    try:
                        vector = vector - val * self.utri[row_idx, :]
                    except IndexError:
                        print(self.utri)
                        print(col_idx)
                        print(vector)
                        print(self.pivot_lookup)
                        raise
                elif current_pivot == len(input_vector) + 1:
                    current_pivot = col_idx

        if current_pivot < len(vector)-1:
            return Constr.VALID, current_pivot, vector
        elif current_pivot == len(vector)-1:
            return Constr.INFEASIBLE, None, None
        else:
            return Constr.REDUNDANT, None, None

    def add_constraint(self, paths, edge):
        # Convert to constraint row
        flow = edge[1]
        row = np.array([0] * self.instance.k + [flow])
        for i in paths:
            row[i] = 1

        # Early-out: if the flow-value of this edge is smaller
        # than the number of paths across is, no integral solution
        # can exist.
        if len(paths) > flow:
            return None

        dependence_flag, pivot_idx, reduced_row = self._check_lin_dep(row)

        if dependence_flag == Constr.REDUNDANT:
            return self
        elif dependence_flag == Constr.INFEASIBLE:
            return None
        # elif dependence_flag == Constr.SOLVED:
        #     return solution  # Instance of SolvedConstr

        assert(dependence_flag == Constr.VALID)

        res = self._copy_with_new_row(row, reduced_row, pivot_idx)
        # print("self.utri", self.utri)
        # print(" res.utri", res.utri)

        # if we've already gotten to a full rank matrix, we can solve
        if res.rank == res.instance.k:
            # COMPUTE WEIGHTS
            # weights = np.linalg.solve(M, b).T
            weights = np.array(list(sorted(res.utri[:, -1])))
            weights = weights.astype(float).tolist()
            # print("Full rank matrix so solving. weights:")
            # print(weights)
            for i, w in enumerate(weights):
                if w <= 0 or not w.is_integer():
                    return None
                weights[i] = int(w)
            # print("Weights are {}".format(weights))
            return SolvedConstr(weights, res.instance)

        # Keep track of path-weights that are determined already.
        if len(paths) == 1:
            res.known_values[paths[0]] = flow

        return res

    def route_cycles_and_satisfy_subpath_constraints(self, og_graph,
                                                     stats_out):
        """Add in the subpath constraints and see if this set of constraints
        has a solution."""
        # see which weights are fixed
        fixed = dict()
        for i, row in enumerate(self.utri):
            sorted_a = np.sort(row[:-1])
            if sorted_a[-1] == 1 and sorted_a[-2] == 0:
                fixed[np.where(row[:-1] == 1)[0][0]] = int(row[-1])
        max_weight = self.instance.flow - sum(fixed.values())
        num_unfixed = self.utri.shape[0] - len(fixed)
        weight_ranges = [list(range(1, max_weight + 2 - num_unfixed))
                         for x in range(self.instance.k)
                         if x not in fixed][:-1]
        # print("fixed is", fixed)
        # print("weight ranges is", weight_ranges)
        # create a grid of weight options to try
        grid = np.array(np.meshgrid(*weight_ranges)).\
            T.reshape(-1, len(weight_ranges))
        for row in grid:
            # create a SolvedConstr instance and see if it satisfies subpath
            # constraints
            row = list(row)
            weights = [None] * self.instance.k
            remaining_flow = self.instance.flow
            for w in fixed:
                weights[w] = fixed[w]
                remaining_flow -= fixed[w]
            remaining_flow -= sum(row)
            for i in range(len(weights)):
                if not row and weights[i] is None:
                    weights[i] = remaining_flow
                elif weights[i] is None:
                    weights[i] = row.pop()
            # print("fixed is", fixed)
            # print(weights)
            # don't even try to solve if weights to don't sum to flow or
            # contains a non positive value
            if sum(weights) != self.instance.flow:
                return False
            if len([x for x in weights if x < 1]) > 0:
                return False
            res = SolvedConstr(weights, self.instance).\
                route_cycles_and_satisfy_subpath_constraints(og_graph,
                                                             stats_out)
            if res:
                return res

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # Since we keep A sorted the following comparisons work.
        A_pivots = filter(lambda x: x > -1, self.pivot_lookup)
        B_pivots = filter(lambda x: x > -1, other.pivot_lookup)
        A = self.utri[list(A_pivots), :]
        B = other.utri[list(B_pivots), :]
        if not np.array_equal(A, B):
            return False

        # This should always be true in our application, added
        # only for completeness.
        if self.instance != other.instance:
            return False

        return True

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return self.hashvalue


class SolvedConstr:
    """
    Special case of a constraint in which all path-weights have
    already been determined.
    """
    def __init__(self, path_weights, instance):
        self.instance = instance
        self.path_weights = tuple(path_weights)

        self.hashvalue = hash(self.path_weights) ^ hash((self.instance.k,
                                                         self.instance.flow))
        self.rank = self.instance.k

    def __repr__(self):
        return "SolvedConstr " + str(self.path_weights)

    def is_redundant(self):
        # We can reduce the number of redundant solutions by imposing that
        # the path-weights are sorted.
        if not all(self.path_weights[i] <= self.path_weights[i+1]
                   for i in range(self.instance.k-1)):
            return True

        # Also ensure that the guess/bound on the max weight is honored
        if self.path_weights[-1] < self.instance.max_weight_bounds[0] or \
                self.path_weights[-1] > self.instance.max_weight_bounds[1]:
            return True
        return False

    def _test_row(self, row):
        # Check whether row is compatible with path weights
        flow = row[-1]
        summed = np.dot(row[:-1], self.path_weights)

        if flow != summed:
            return Constr.INFEASIBLE, None

        return Constr.SOLVED, self

    def add_constraint(self, paths, edge):
        # Test whether constraint is compatible
        summed = 0
        for i in paths:
            summed += self.path_weights[i]
        if summed == edge[1]:
            return self
        else:
            return None

    def get_replacement_pathset(self, pathset, result):
        routing, indices, in_edges = result
        print("#### Incorporating routing through scc:", result[0])
        print("#### in edges", in_edges)
        replacement_pathset = []
        for i, path in enumerate(pathset):
            print("###### Path ", i, path)
            try:
                routing_to_insert = routing[indices.index(i)]
                print("###### routing to insert",
                      routing_to_insert)
                print("###### in edges", in_edges)
                in_edge = list(set(in_edges) & set(path[0]))[0]
                print("###### in edge", in_edge)
                in_edge_index = path[0].index(in_edge)
                print("###### in edge index", in_edge_index)
                first_half = path[0][:in_edge_index + 1]
                print("###### first half", first_half)
                second_half = path[0][in_edge_index + 1:]
                print("###### second half", second_half)
                new_path = first_half + \
                    tuple(routing_to_insert) + second_half
                replacement_pathset.append((new_path, path[1]))
            except (ValueError, IndexError):
                # this path enters and leaves by the same node
                # so it doesn't need to be changed (ValueError)
                # or this path doesn't even go through the SCC
                # (IndexError)
                print("###### path does not need to be changed")
                replacement_pathset.append(path)
        print("###### After routing through cycles, pathset is",
              replacement_pathset)
        return replacement_pathset

    def route_cycles_and_satisfy_subpath_constraints(self, graph, stats_out):
        """Add in the subpath constraints and see if this set of constraints
        has a solution."""
        start_time = time.time()
        solution_pathsets_all = recover_paths(self.instance, self.path_weights)
        print("\nProcessing a solved constraint system with {} pathsets.".
              format(len(solution_pathsets_all)))
        # only look into cycles if there are cycles
        sol_pathsets = []
        if len([x for x in self.instance.sccs if len(x) > 1]) == 0:
            print("No cycles to process.")
            sol_pathsets = solution_pathsets_all
        else:
            counter = 1
            for pathset in solution_pathsets_all:
                this_pathset_sols = [pathset]
                print("this_pathset_sols is", this_pathset_sols)
                print("## Trying to route pathset {}, {}, over cycles".format(
                    counter, pathset))
                counter += 1
                cycle_counter = 0
                for c in [x for x in self.instance.sccs if len(x) > 1]:
                    cycle_counter += 1
                    print("Processing cycle number", cycle_counter)
                    valid_routings = self.instance.cyclic_graph.\
                        route_cycle(c, self.instance.graph, pathset, stats_out)
                    if valid_routings:
                        # this scc is can be covered by this pathest
                        # incorporate into path
                        new_sol_pathsets = []
                        for result in valid_routings:
                            for p in this_pathset_sols:
                                replacement_pathset = self.\
                                    get_replacement_pathset(p, result)
                                new_sol_pathsets.append(replacement_pathset)
                                print("There are {} working pathsets".format(
                                    len(new_sol_pathsets)))
                        this_pathset_sols = new_sol_pathsets
                    else:
                        # this pathset doesn't work, so stop considering it
                        print("## Pathset doesn't work for cycle")
                        break
                else:  # executes if we processed all sccs successfully
                    print("Successfully processed all cycles.")
                    sol_pathsets.extend(this_pathset_sols)

        if sol_pathsets:
            print("There are {} possible solutions. ".format(len(sol_pathsets))
                  + "Now check if any satisfy the subpath constraints.")
        print("It took {} seconds to find valid routings through sccs".
              format(time.time() - start_time))
        for sol_pathset in sol_pathsets:
            print("## Checking", sol_pathset)
            # at this point, we've got paths in the reduced graph, not the scc
            # graph. so we should switch self.instance.graph to be
            # self.instance.cyclic_graph.
            self.instance.graph = self.instance.cyclic_graph

            # convert contracted paths to full paths
            weight_vec = []
            paths = []
            for path_deq, weight in sol_pathset:
                real_path = []
                for arc in path_deq:
                    real_path.extend(self.instance.graph.mapping[arc])
                node_seq = [graph.source()]
                for arc in real_path:
                    node_seq.append(graph.arc_info[arc]['destin'])
                weight_vec.append(weight)
                paths.append(node_seq)
            print("recovered paths are", paths)

            # there might not be any subpath constraints
            if not self.instance.graph.subpath_constraints:
                print("No subpath constraints")
                return paths, weight_vec
            else:
                print("There are {} subpath constraints to check".
                      format(len(self.instance.graph.subpath_constraints)))
                for (L, d) in zip(self.instance.graph.subpath_constraints,
                                  self.instance.graph.subpath_demands):
                    total_coverage = 0
                    print("constraint is", L)
                    print("demand is", d)
                    for (path, weight) in zip(paths, weight_vec):
                        if str(L)[1:-1] in str(path)[1:-1]:
                            total_coverage += weight
                    # if we don't meet demand, this solution paths is not good
                    print("total coverage is", total_coverage)
                    if d > total_coverage:
                        print("doesn't meet subpath demand")
                        break
                # executes if all subpath constraint satisfied
                else:
                    return paths, weight_vec

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.instance != other.instance:
            return False

        return self.path_weights == other.path_weights

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return self.hashvalue


class PathConf:
    """Class representing paths ending in a set of vertices."""

    def __init__(self, vertex=None, paths=None):
        # paths is a dict mapping from vertex to set of paths crossing it
        self.paths = {}
        # arcs_used is a dict mapping path to arc it crossed to get to vertex
        self.arcs_used = {}
        if vertex is not None:
            self.paths[vertex] = frozenset(paths)
            for path in paths:
                self.arcs_used[path] = -1

    def copy(self):
        res = PathConf()
        for v, paths in self.paths.items():
            res.paths[v] = frozenset(paths)
        res.arcs_used = {}
        for key, val in self.arcs_used.items():
            res.arcs_used[key] = val
        return res

    def __iter__(self):
        return iter(self.paths)

    def __contains__(self, v):
        return v in self.paths

    def __getitem__(self, v):
        return self.paths[v]

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        res = "PathConf("
        for v in self.paths:
            res += "{}: {}, ".format(v, list((p, self.arcs_used[p]) for p in
                                             self.paths[v]))
        return res[:-2] + ")"

    def __eq__(self, other):
        # print("Checking equality of two PathConf objects.")
        # print("First has paths={}".format(self.paths))
        # print("Second has paths={}".format(self.paths))
        # print("First has arcs_used={}".format(self.arcs_used))
        # print("Second has arcs_used={}".format(other.arcs_used))
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self.paths) != len(other.paths):
            return False
        for v in self.paths:
            if v not in other.paths:
                return False
            if self.paths[v] != other.paths[v]:
                # print("They are not equal.")
                return False
        if self.arcs_used != other.arcs_used:
            # print("They are not equal.")
            return False
        # print("They are equal.")
        return True

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset(self.paths.items()))

    def push(self, v, edges):
        """
        Return all ways that the paths ending in v can be 'pushed' along the
        provided edges.
        """
        if len(self.paths[v]) < len(edges):
            # There are fewer paths ending in v than out-arcs.
            # This cannot be extended to a solution.
            return
        if len(self.paths[v]) == 0:
            raise ValueError("{} has no paths running through it.".format(v))
        if len(edges) == 0:
            raise ValueError("{} has no edges exiting it".format(v))
        for path_distribution in distribute(self.paths[v], edges):
            res = self.copy()  # Copy old paths,
            for p in res.paths[v]:
                del res.arcs_used[p]  # remove paths ending in v
            del res.paths[v]
            # Push paths over prescribed edges
            for arc, pathset in path_distribution:
                tail, _, arc_label = arc
                if tail in res.paths:
                    res.paths[tail] = frozenset(res.paths[tail] | set(pathset))
                else:
                    res.paths[tail] = frozenset(pathset)
                for path in pathset:
                    res.arcs_used[path] = arc_label

            yield res, path_distribution


def distribute(paths, edges):
    """
    Distribute n paths on k edges such that every edge has at least one path.
    """
    k = len(edges)
    paths = list(paths)
    for partition in algorithm_u(paths, k):
        # Generate non-empty partitions with k blocks
        for perm in itertools.permutations(partition):  # Permute to distribute
            yield list(zip(edges, perm))


def recover_paths(instance, weights, silent=True):
    """Recover all paths that correspond to the weights given."""
    graph = instance.graph
    k = instance.k

    if not silent:
        print(graph)

    # since we know all the weights, we can stored them as a solved constraint
    # system
    globalconstr = SolvedConstr(weights, instance)

    assert len(weights) == k
    allpaths = frozenset(range(k))

    old_paths = [[PathConf(graph.source(), allpaths)]]

    for v in instance.ordering[:-1]:
        new_paths = []
        for path in old_paths:
            # get last item in this path
            last_assignment = path[-1]
            for pushed_paths, dist in \
                    last_assignment.push(v, graph.labeled_neighborhood(v)):
                # make sure the sets of paths that were pushed along each
                # edge sum to the proper flow value.  If not, don't create a
                # new table entry for this path set.
                # print("Trying", pushed_paths)
                for arc, pathset in dist:  # Paths pathset coincide on arc
                    success = globalconstr.add_constraint(pathset, arc)
                    if success is None:
                        break
                # we found a way to extend this path. add the extension
                # to the list of paths.
                else:
                    path_copy = path.copy()
                    path_copy.append(pushed_paths)
                    new_paths.append(path_copy)

        old_paths = new_paths
        # print("at node {} we have {} paths".format(v, len(old_paths)))

    # recover the paths
    # we want a list of lists of [path deque, weight]
    full_paths = [[[deque(), weight] for weight in weights] for path in
                  new_paths]
    for i, path in enumerate(new_paths):
        # print("Path is:", path)
        for conf in path:
            for v in conf:
                # incidence is the list of all paths passing through this vert
                incidence = conf[v]
                # vertices might repeat in consecutive table entries if an edge
                # is "long" wrt the topological ordering.  Don't add it twice
                # to the path lists in this case.
                for p in incidence:
                    arc_used = conf.arcs_used[p]
                    if arc_used == -1:
                        break
                    if len(full_paths[i][p][0]) == 0 or \
                            full_paths[i][p][0][-1] != arc_used:
                        full_paths[i][p][0].append(arc_used)

    # print("length of full paths is", len(full_paths))
    for sol in full_paths:
        for path in sol:
            path[0] = tuple(path[0])
    sols = []
    for x in full_paths:
        x = [tuple(y) for y in x]
        sols.append(x)
    sols = list(set([tuple(sorted(x)) for x in sols]))
    # print("len of deduped paths is", len(sols))
    return sols
