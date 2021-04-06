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
import ifd_package.flows.ifd as ifd
from os import path
from os import mkdir
import sys
from coaster.guess_weight import solve
from coaster.parser import read_instances
from coaster.flow import Instance
from coaster.graphs import test_flow_cover
sys.setrecursionlimit(2000)


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


def find_exact_sol(instance, maxtime, max_k, stats_out):
    """
    Find an optimal flow decomposition (paths and weights) for instance.
    stats_out is a file for writing stats about this instance.

    This is the main function for running the FPT version of Coaster.
    """

    if maxtime is None:
        maxtime = -1
    print("Searching for minimum-sized set of weights, timeout set at {}"
          "".format(maxtime))
    try:
        with timeout(seconds=maxtime):
            while True:
                # if we're just running through large k vals, stop
                if max_k:
                    assert instance.k <= max_k
                print("\n# \tTrying to solve with k = {}".format(instance.k))
                solution = solve(instance, graph, stats_out, silent=False)
                if bool(solution):
                    break
                instance.try_larger_k()
            elapsed = time.time() - start
            print("\n# Solution time was {:.2f} seconds".format(elapsed))
            return solution, elapsed
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set(), maxtime
    except AssertionError:
        print("Reached max k after {} seconds".format(elapsed))
        return set(), elapsed


def find_heuristic_sol(graph, maxtime,):
    """
    Find a flow decomposition for instance. NOTE: unsure whether this works for
    cyclic instances. TODO: add statsfile. TODO: can we use the reduced graph?

    This is the main function for running the heuristic version of Coaster.
    """

    if maxtime is None:
        maxtime = -1
    print("Searching for heuristic solution. Timeout set at {}"
          "".format(maxtime))
    try:
        with timeout(seconds=maxtime):
            ifd_instance = ifd.InexactFlowInstance(
                graph.get_mifd_reduction())
            ifd_instance.solve()
            ifd_instance.graph.convert_paths()
            paths = ifd_instance.graph.get_converted_paths()
            weights = ifd_instance.graph.get_weights()
            elapsed = time.time() - start
            print("\n# Solution time was {:.2f} seconds".format(elapsed))
            return (paths, weights), elapsed
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set(), maxtime
    except TypeError:
        print("TypeError in graph")
        return set(), 0


if __name__ == "__main__":
    """
        Main script
    """
    # parse arguments
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
    parser.add_argument('--print_arcs', help="Make output include arc labels.",
                        action='store_true')
    parser.add_argument('--print_contracted', help="Print contracted graph.",
                        action='store_true')
    parser.add_argument('--experiment_info', help='Print out experiment-'
                        'relevant info in format convenient for processing.',
                        action='store_true')
    parser.add_argument("--create_graph_pics", help="Generate PDF of graphs",
                        action='store_true')
    parser.add_argument("--max_k", help="Largest k to consider for any graph",
                        type=int)
    parser.add_argument('--heuristic', default=False, action='store_true')

    args = parser.parse_args()

    graph_file = args.file
    filename = path.basename(graph_file)
    truth_file = "{}.truth".format(path.splitext(graph_file)[0])
    stats_file = "stats_files/" + graph_file.split("/")[-1] + "_stats.txt"
    if not path.isdir("stats_files"):
        mkdir("stats_files")
    stats_out = open(stats_file, "w")
    stats_out.write("filename,graphname,n,m,contracted_n,contracted_m," +
                    "scc_n,scc_m,num_cycles,size_of_cycles...,routings_" +
                    "over_cycle...,\n")

    maxtime = args.timeout
    if maxtime:
        print("# Timeout is set to", maxtime)
    else:
        print("# No timeout set")

    max_k = args.max_k
    if max_k:
        print("# Max k is set to", max_k)
    else:
        print("# No max_k set")

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
        truth_file = None

    # Iterate over every graph-instance inside the input file
    for graphdata, k, index in read_instances(graph_file, truth_file):
        graph, graphname, graphnumber = graphdata

        if args.create_graph_pics:
            graph.show("graph_pics/graph{}.pdf".format(graphnumber))

        if instances and index not in instances:
            continue
        print("This is graph index {}".format(index))
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
        # graphname is graphnumber-fileindex in our instances, so it can be
        # used to join output with groundtruth
        stats_out.write("{},{},{},{},".format(filename, graphname, n_input,
                                              m_input))

        if args.print_contracted:
            print("Original graph is:")
            graph.print_out()
        graph.write_graphviz("original_graph.dot")
        graph.check_flow()
        start = time.time()
        # contract in-/out-degree 1 vertices
        reduced, mapping = graph.contracted()
        # reduced is the graph after contractions;
        # mapping enables mapping paths on reduced back to paths in graph
        stats_out.write("{},{},".format(reduced.num_nodes(),
                                        reduced.num_edges()))
        if args.print_contracted:
            print("Contracted graph is:")
            reduced.print_out()
            print("contraction mapping is,", mapping)

        # create a graph with all strongly connected components contracted to
        # single vertices
        scc_reduced, sccs = reduced.scc()
        stats_out.write("{},{},{},".format(
            scc_reduced.num_nodes(), scc_reduced.num_edges(), len(sccs) - 2))
        for scc in sccs[1:-1]:
            stats_out.write("{},".format(len(scc)))
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
            if args.heuristic:
                solution, time_weights = find_heuristic_sol(graph, maxtime)
            else:
                k_start = 1
                instance = Instance(scc_reduced, k_start, reduced, sccs)
                k_improve = instance.best_cut_lower_bound
                print("# Reduced instance has n = {}, m = {}, and lower_bound "
                      "= {}:".format(n, m, instance.k), flush=True)

                k_cutset = instance.max_edge_cut_size  # this is for reporting
                solution, time_weights = find_exact_sol(instance, maxtime,
                                                        max_k, stats_out)
            if solution:
                stats_out.write("{}".format(len(solution[0])))
            else:
                stats_out.write("{}".format(0))

            # recover the paths in an optimal solution
            if bool(solution):
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

        # print experimental statistics
        if args.experiment_info:
            print("# All_info\tn_in\tm_in\tn_red\tm_red\tk_gtrue\tk_cut"
                  "\tk_impro\tk_opt\ttime_w\ttime_p")
            print("All_info\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                  "".format(n_input, m_input, n, m, k_gtrue, k_cutset,
                            k_improve, k_opt, time_weights,
                            ))
            print("weights\t", *[w for w in weights])
        stats_out.write("\n")
        print("Finished instance.\n")
    stats_out.close()
