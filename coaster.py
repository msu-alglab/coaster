#! /usr/bin/env python3
#
# This file is part of Coaster, https://github.com/msualglab/Coaster/,
# and is Copyright (C) Montana State University, 2021. It is licensed
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
from pathlib import Path
import sys
from coaster.guess_weight import solve
from coaster.parser import read_instances
from coaster.flow import Instance
from coaster.graphs import test_flow_cover
import coaster.fd as fd
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
        self.start_pro_time = time.process_time()

    def handle_timeout(self, signum, frame):
        print("cpu time is", time.process_time() - self.start_pro_time)
        raise TimeoutError(self.error_message)

    def handle_alarm(self, signum, frame):
        cpu_time = time.process_time() - self.start_pro_time
        if cpu_time > self.seconds:
            raise TimeoutError(self.error_message)
        else:
            signal.signal(signal.SIGALRM, self.handle_alarm)
            signal.alarm(1)

    def __enter__(self):
        if self.seconds > 0:
            # signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.signal(signal.SIGALRM, self.handle_alarm)
            signal.alarm(1)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)
        print("cpu time was", time.process_time() - self.start_pro_time)


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


def find_exact_sol(instance, maxtime, max_k):
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
                solution = solve(instance, graph, silent=True)
                if bool(solution):
                    break
                instance.try_larger_k()
            elapsed = time.time() - start
            cpu_elapsed = time.process_time() - start_cpu
            print("\n# Solution time was {:.2f} seconds".format(elapsed))
            return solution, cpu_elapsed
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set(), maxtime
    except AssertionError:
        print("Reached max k after {} seconds".format(elapsed))
        return set(), elapsed


def find_fd_heuristic_sol(graph, maxtime, no_br=False):
    """
    Find a flow decomposition for instance.

    This is the main function for running the FD heuristic version of Coaster.
    """

    if maxtime is None:
        maxtime = -1
    print("Searching for FD heuristic solution. Timeout set at {}"
          "".format(maxtime))
    try:
        with timeout(seconds=maxtime):
            fd_instance = fd.ExactFlowInstance(graph)
            # by default, we solve with bridge reweighting. but if no_br=True,
            # don't use bridge reweighting
            fd_instance.solve(no_br)
            fd_instance.convert_paths()
            # fd_instance.splice_heuristic()
            paths = fd_instance.paths
            weights = fd_instance.weights
            elapsed = time.time() - start
            cpu_elapsed = time.process_time() - start_cpu
            print("\n# Solution time was {:.2f} seconds".format(elapsed))
            return (paths, weights), cpu_elapsed
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set(), maxtime
    # catching this type error exception was for ANNs, but we may need some
    # sort of similar thing so keeping the code here in case
    # except TypeError:
        # print("TypeError in graph")
        # return set(), 0


if __name__ == "__main__":
    """
        Main script
    """
    # start overall timer
    overall_start_cpu = time.process_time()
    overall_start_wallclock = time.perf_counter()

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
    parser.add_argument('--fd_heuristic', default=False, action='store_true')
    parser.add_argument('--fd_heuristic_no_br', default=False,
                        action='store_true')

    args = parser.parse_args()

    graph_file = args.file
    filename = path.basename(graph_file)
    filenum = graph_file.split(".graph")[0].split("graphs/sc")[1]
    print("filenum is", filenum)
    truth_file = "{}.truth".format(path.splitext(graph_file)[0])
    # assume that file structure is some_dir/experiment_info/truth/truthfile,
    # some_dir/experiment_info/graphs/graphfiles, and we want to put the output
    # in some_dir/experiment_info/predicted_exp_type/pred.txt
    if args.fd_heuristic:
        exp_type = "fd_heur"
    elif args.fd_heuristic_no_br:
        exp_type = "fd_heur_no_br"
    else:
        exp_type = "fpt"
    pred_path_filename = Path(graph_file).parents[1] /\
        ("predicted_" + exp_type) / ("pred" + filenum + ".txt")
    try:
        pred_path_filename.parents[0].mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    # make a runtime output filename as well
    runtime_filename = Path(graph_file).parents[1] /\
        ("runtimes_" + exp_type) / ("runtimes" + filenum + ".txt")
    try:
        runtime_filename.parents[0].mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

    output = open(pred_path_filename, "w")
    runtime_output = open(runtime_filename, "w")

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

        if args.print_contracted:
            print("Original graph is:")
            graph.print_out()
        graph.write_graphviz("original_graph.dot")
        graph.check_flow()
        start = time.time()
        start_cpu = time.process_time()
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
        # TODO: REMOVE
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
        # non-trivial, so solve
        else:
            if args.fd_heuristic:
                solution, time_weights = find_fd_heuristic_sol(graph, maxtime)
            elif args.fd_heuristic_no_br:
                solution, time_weights = find_fd_heuristic_sol(graph, maxtime,
                                                               no_br=True)
            else:
                k_start = 1
                instance = Instance(scc_reduced, k_start, reduced, sccs)
                k_improve = instance.best_cut_lower_bound
                print("# Reduced instance has n = {}, m = {}, and lower_bound "
                      "= {}:".format(n, m, instance.k), flush=True)

                k_cutset = instance.max_edge_cut_size  # this is for reporting
                solution, time_weights = find_exact_sol(instance, maxtime,
                                                        max_k)
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
                output.write("# graph number = {} name = {}\n".
                             format(graphnumber, graphname))
                runtime_output.write("{} {}\n".format(graphname, time_weights))
                for p, weight in zip(paths, weights):
                    output.write(" ".join([str(x) for x in [weight] + p]) +
                                 "\n")

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
    output.close()
    runtime_output.close()
    # report overall times
    print("Overall cpu time: {:2f} seconds".format
          (time.process_time() - overall_start_cpu))
    print("Overall wall clock time: {:2f} seconds".format
          (time.perf_counter() - overall_start_wallclock))
