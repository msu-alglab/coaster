#! /usr/bin/env python3
#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# -*- coding: utf-8 -*-
# python libs
import time
import argparse
import signal
# import cProfile
# local imports
from os import path
import sys
from cyclic_flows.guess_weight import solve
from cyclic_flows.parser import read_instances
from cyclic_flows.flow import Instance
from cyclic_flows.graphs import test_flow_cover


# Override error message to show help message instead
class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.exit(2)


# Timeout context, see
#   http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class timeout:
    """
    Enable a timeout for a function call.

    Used to skip an input graph-instance once our algorithm has run for a
    specified amount of time without terminating.
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)


def index_range(raw):
    """Parse index ranges separated by commas e.g. '1,2-5,7-11'."""
    if not raw:
        return None

    indices = set()
    try:
        with open(raw, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == '#':
                    continue
                if " " in line:
                    line = line.split()[0]
                indices.add(int(line))
        return indices
    except FileNotFoundError:
        pass

    for s in raw.split(","):
        if "-" in s:
            start, end = s.split("-")
            indices.update(range(int(start), int(end)+1))
        else:
            indices.add(int(s))
    return indices


def find_opt_size(instance, maxtime):
    """Find the optimum size of a flow decomposition."""
    if maxtime is None:
        maxtime = -1
    print("Searching for minimum-sized set of weights, timeout set at {}"
          "".format(maxtime))
    try:
        with timeout(seconds=maxtime):
            while True:
                print("\n# \tTrying to solve with k = {}".format(instance.k))
                solution = solve(instance, graph, silent=True)
                if bool(solution):
                    break
                instance.try_larger_k()
            elapsed = time.time() - start
            print("\n# Solution time was {:.2f} seconds".format(elapsed))
            return solution, elapsed
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set(), maxtime


if __name__ == "__main__":
    """
        Main script
    """
    readme_filename = path.join(path.dirname(__file__), 'readme_short.txt')
    with open(readme_filename, 'r') as desc:
        description = desc.read()
    parser = DefaultHelpParser(description=description)
    parser.add_argument('file',
                        help='A .graph file containing the input graph(s).')
    parser.add_argument('--indices', help='Either a file containing indices '
                        '(position in .graph file) on which to run, '
                        'or a list of indices separated by commas. '
                        'Ranges are accepted as well, e.g. "1,2-5,6".',
                        type=index_range)
    parser.add_argument('--timeout',
                        help='Timeout in seconds, after which execution'
                        ' for a single graph will be stopped.', type=int)
    parser.add_argument('--skip_truth', help="Do not check for *.truth."
                        " Instead, start from our computed lower-bound on k.",
                        action='store_true')
    parser.add_argument('--print_arcs', help="Make output include arc labels.",
                        action='store_true')
    parser.add_argument('--print_contracted', help="Print contracted graph.",
                        action='store_true')
    parser.add_argument('--experiment_info', help='Print out experiment-'
                        'relevant info in format convenient for processing.',
                        action='store_true')
    parser.add_argument("--no_recovery", help="Only print the number of paths"
                        " and their weights in an optimal decomposition, but"
                        " do not recover the paths.", action='store_true')
    parser.add_argument("--create_graph_pics", help="Generate PDF of graphs",
                        action='store_true')

    args = parser.parse_args()

    graph_file = args.file
    filename = path.basename(graph_file)
    truth_file = "{}.truth".format(path.splitext(graph_file)[0])

    maxtime = args.timeout
    if maxtime:
        print("# Timeout is set to", maxtime)
    else:
        print("# No timeout set")

    recover = not args.no_recovery
    if recover:
        print("# Recovering paths")
    else:
        print("# Only printing path weights")

    instances = args.indices
    if instances:
        a = sorted(list(instances))
        res = str(a[0])
        lastprinted = a[0]
        for current, last in zip(a[1:], a[:-1]):
            if current == last+1:
                continue
            if last != lastprinted:
                res += "-"+str(last)
            res += ","+str(current)
            lastprinted = current

        if lastprinted != a[-1]:
            res += "-" + str(a[-1])

        print("# Running on instance(s)", res)
    else:
        print("# Running on all instances")

    if path.isfile(truth_file):
        print("# Ground-truth available in file {}".format(truth_file))
    else:
        print("# No ground-truth available. Guessing parameter.")
        truth_file = None

    # Iterate over every graph-instance inside the input file
    for graphdata, k, index in read_instances(graph_file, truth_file):
        graph, graphname, graphnumber = graphdata

        if args.create_graph_pics:
            graph.show("graph_pics/graph{}.pdf".format(graphnumber))

        if instances and index not in instances:
            continue
        n_input = len(graph)
        m_input = len(list(graph.edges()))
        k_gtrue = k if k else "?"
        k_opt = None
        k_improve = 0
        weights = []
        time_weights = None
        time_path = None
        print("\nFile {} instance {} name {} with n = {}, m = {}, and truth = "
              "{}:".format(filename, graphnumber, graphname, n_input,
                           m_input, k if k else "?"), flush=True)

        if args.print_contracted:
            print("Original graph is:")
            graph.print_out()
        start = time.time()
        # contract in-/out-degree 1 vertices
        reduced, mapping = graph.contracted()
        # reduced is the graph after contractions;
        # mapping enables mapping paths on reduced back to paths in graph
        if args.print_contracted:
            print("Contracted graph is:")
            reduced.print_out()
            print("contraction mapping is,", mapping)

        # create a graph with all strongly connected components contracted to
        # single vertices
        scc_reduced, sccs = reduced.scc()
        if args.print_contracted:
            print("sccs are", sccs)
            print("SCC graph is:")
            scc_reduced.print_out()

        n = len(scc_reduced)
        m = len(list(scc_reduced.edges()))

        if len(scc_reduced) <= 1:
            print("Trivial.")
            # continue
            k_improve = 1
            k_opt = 1
            k_cutset = 1
            time_weights = 0
            if m_input != 0:
                weights = [list(graph.edges())[0][2]]
            else:
                weights = [0]
        else:
            # create an instance of the graph
            if args.skip_truth:
                k = 1
            instance = Instance(scc_reduced, k, reduced, sccs)
            k_improve = instance.best_cut_lower_bound
            print("# Reduced instance has n = {}, m = {}, and lower_bound "
                  "= {}:".format(n, m, instance.k), flush=True)

            k_cutset = instance.max_edge_cut_size

            solution, time_weights = find_opt_size(instance, maxtime)

            # recover the paths in an optimal solution
            if bool(solution) and recover:
                paths, weights = solution
                start_path_time = time.time()
                print("# Solutions:")
                weight_vec = []
                k_opt = len(weights)

                for p, w in zip(paths, weights):
                    print("{}: {}".format(w, p))
                # Check solution:
                test_flow_cover(graph, paths, weights)
                print("# Paths, weights pass test: flow decomposition"
                      " confirmed.")
                # Print solutions

        # print experimental statistics
        if args.experiment_info:
            print("# All_info\tn_in\tm_in\tn_red\tm_red\tk_gtrue\tk_cut"
                  "\tk_impro\tk_opt\ttime_w\ttime_p")
            print("All_info\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                  "".format(n_input, m_input, n, m, k_gtrue, k_cutset,
                            k_improve, k_opt, time_weights,
                            ))
            print("weights\t", *[w for w in weights])
        print("Finished instance.\n")
