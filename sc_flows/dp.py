#! /usr/bin/env python3
#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# python libs
from collections import defaultdict, deque

# local imports
from sc_flows.flow import Constr, SolvedConstr, PathConf


def solve(instance, og_graph, silent=True, guessed_weights=None):
    """
    Find a feasible set of weights consistent with guessed_weights.

    guessed_weights is a sorted tuple of size k in which any of the entries
    might be None.  If guessed_weights is k integers, check whether this
    solution works.  If any entry of guessed_weights is None, check whether a
    solution exists with each None replaced by a integer that respects the
    sorted order of the tuple.
    """
    graph = instance.graph
    k = instance.k

    # The only constraint a priori is the total flow value
    globalconstr = Constr(instance)
    guessed_weights = [None]*k if not guessed_weights else guessed_weights

    assert len(guessed_weights) == k
    for index, guess in enumerate(guessed_weights):
        if guess is None:
            continue
        globalconstr = globalconstr.add_constraint([index], (0, guess))
        if globalconstr is None:
            return set()

    # Build first DP table
    old_table = defaultdict(list)
    allpaths = frozenset(range(k))
    # All k paths `end' at source
    old_table[PathConf(graph.source(), allpaths)] = list([globalconstr])

    # run dynamic progamming
    for v in instance.ordering[:-1]:
        if not silent:
            print("Processing node {}".format(v))
            print("In the old table, all path mappings are:")
            print(old_table.keys())
            print("In the old table, all constraints are:")
            print(old_table.values())
        new_table = defaultdict(list)
        # paths is a PathConf object. constraints is a Constr object.
        for paths, constraints in old_table.items():
            # Distribute paths incoming to i onto its neighbours
            if not silent:
                print("  Pushing {}".format(paths))
            for newpaths, dist in paths.push(v, graph.labeled_neighborhood(v)):
                # debug_counter = 0
                # for this choice of path extension, look at each L so far and
                # see if adding in this path extension yields a feasible new L
                for constr in constraints:
                    # print("  Combining w/ constraints", constraints)
                    # Update constraint-set constr with new constraints imposed
                    # by our choice of pushing paths o..........es (as
                    # described by dist)
                    curr_constr = constr
                    try:
                        for e, p in dist:  # Paths p coincide on edge e
                            updated_constr = curr_constr.add_constraint(p, e)
                            if updated_constr is not None:
                                curr_constr = updated_constr
                            else:
                                break
                        # this is only done if we did not hit the break. that
                        # is, if all constraints were feasible.
                        else:
                            # WORRY ABOUT THIS EVENTUALLY
                            if constr.rank == curr_constr.rank or \
                                    not curr_constr.is_redundant():
                                # Add to DP table
                                if not silent:
                                    print("Adding an extension to table.")
                                    print("Specifically, adding ({}, {})".
                                          format(newpaths, curr_constr))
                                new_table[newpaths].append(curr_constr)
                    except ValueError as err:
                        print("Problem while adding constraint", p, e)
                        raise err
        # replace tables
        old_table = new_table

    # At the end, we may have some SolvedConstr objects in the table and some
    # Constr objects, which are not full rank.
    for key in new_table:
        for sys in new_table[key]:
            # we need to pass in the original graph in order to recover paths
            # print("\nending constr is:", sys)
            solution = sys.route_cycles_and_satisfy_subpath_constraints(
                og_graph)
            if sys:
                return solution


def recover_paths(instance, weights, silent=True):
    """Recover the paths that correspond to the weights given."""
    graph = instance.graph
    k = instance.k

    if not silent:
        print(graph)

    # since we know all the weights, we can stored them as a solved constraint
    # system
    globalconstr = SolvedConstr(weights, instance)

    assert len(weights) == k
    allpaths = frozenset(range(k))

    # Build DP table, which will map path configurations to the path
    # configurations in the previous tables that yielded the specific entry
    # All k paths start at the source vertex
    initial_entries = {PathConf(graph.source(), allpaths): None}
    backptrs = [initial_entries]

    # Run DP
    for v in instance.ordering[:-1]:
        entries = {}
        for old_paths in backptrs[-1].keys():
            # Distribute paths incoming to i onto its neighbors
            if not silent:
                print("  Pushing {}".format(old_paths))
            for new_paths, dist in \
                    old_paths.push(v, graph.labeled_neighborhood(v)):
                # make sure the sets of paths that were pushed along each
                # edge sum to the proper flow value.  If not, don't create a
                # new table entry for this path set.
                for arc, pathset in dist:  # Paths pathset coincide on arc
                    success = globalconstr.add_constraint(pathset, arc)
                    if success is None:
                        break
                else:
                    if new_paths not in entries:
                        entries[new_paths] = old_paths

        # add the new entries to the list of backpointers
        backptrs.append(entries)

    if not silent:
        print("\nDone.")
        for bp in backptrs:
            print(bp)

    # recover the paths
    full_paths = [[deque(), weight] for weight in weights]
    try:
        # the "end" of the DP is all paths passing through the sink
        # conf = PathConf(graph.sink(), allpaths)
        conf = list(backptrs[-1].keys())[0]
        # iterate over the backpointer list in reverse
        for table in reversed(backptrs):
            for v in conf:
                # get the paths crossing this vertex
                incidence = conf[v]
                # vertices might repeat in consecutive table entries if an edge
                # is "long" wrt the topological ordering.  Don't add it twice
                # to the path lists in this case.
                for p in incidence:
                    arc_used = conf.arcs_used[p]
                    if arc_used == -1:
                        break
                    if len(full_paths[p][0]) == 0 or \
                            full_paths[p][0][0] != arc_used:
                        full_paths[p][0].appendleft(arc_used)
            # traverse the pointer backwards
            conf = table[conf]
    except KeyError as e:
        raise Exception("The set of weights is not a valid solution") from e

    return full_paths
