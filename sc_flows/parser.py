#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# local imports
from sc_flows.graphs import AdjList
# python libs
import re
import os

header_regex = re.compile('# graph number = ([0-9]*) name = (.*)')
edge_regex = re.compile('(\d*) (\d*) (\d*\.\d*)') # noqa


def read_sgr(graph_file):
    """Read a single graph from a .sgr file."""
    with open(graph_file, 'r') as f:
        num_nodes = int(f.readline().strip())
        graph = AdjList(graph_file, None, None, num_nodes)
        for line in f:
            edge_data = line.split()
            u = int(edge_data[0])
            v = int(edge_data[1])
            flow = int(float(edge_data[2]))
            graph.add_edge(u, v, flow)
        return graph, None, 0


def enumerate_graphs(graph_file):
    def read_next_graph(f):
        header_line = f.readline()

        if header_line == '':
            return None

        m = header_regex.match(header_line)
        if m is None:
            raise Exception('Misformed graph header line.')
        (graph_number, graph_name) = (m.group(1), m.group(2))

        line = f.readline()
        num_nodes = int(line.strip())

        graph = AdjList(graph_file, graph_number, graph_name, num_nodes)

        while not line == '':
            last_pos = f.tell()
            line = f.readline()

            if line == '':
                break
            # if next line only has one thing in it, we move on to processing
            # subpath constraints
            elif len(line.split()) == 1:
                break

            list = line.split()

            u = int(list[0])
            v = int(list[1])
            flow = int(float(list[2]))

            graph.add_edge(u, v, flow)

        while not line == '':
            last_pos = f.tell()
            line = f.readline()

            if line == '':
                break
            # if next line is a header line, we're done with this graph
            elif line[0] == '#':
                f.seek(last_pos)  # go back to where we were before newline
                break

            list = line.split()

            d = int(float(list[-1]))
            subpath = [int(x) for x in list[:-1]]

            graph.add_subpath_constraint(subpath, d)

        return graph, graph_name, graph_number

    with open(graph_file) as f:
        while True:
            graph_data = read_next_graph(f)
            if graph_data is None:
                break
            else:
                yield graph_data


def enumerate_decompositions(decomposition_file):
    def read_next_decomposition(f):
        header_line = f.readline()

        if header_line == '':
            return None

        m = header_regex.match(header_line)
        if m is None:
            raise Exception('Misformed graph header line.')
        (graph_number, graph_name) = (m.group(1), m.group(2))

        path_decomposition = []
        line = header_line
        while not line == '':
            last_pos = f.tell()
            line = f.readline()

            if line == '':
                break
            elif line[0] == '#':
                f.seek(last_pos)
                break

            l = line.split()
            l = list(map(lambda x: int(x), l))

            path_decomposition.append((l[0], l[1:]))

        return (graph_name, graph_number, path_decomposition)

    with open(decomposition_file) as f:
        while True:
            decomposition = read_next_decomposition(f)
            if decomposition is None:
                break
            else:
                yield decomposition


def read_instances(graph_file, truth_file):
    index = 0
    if os.path.splitext(graph_file)[1] == ".sgr":
        yield read_sgr(graph_file), None, 0
    elif truth_file:
        for graphdata, truthdata in zip(enumerate_graphs(graph_file),
                                        enumerate_decompositions(
                                        truth_file)):
            index += 1
            _, _, solution = truthdata
            yield (graphdata, len(solution), index)
    else:
        for graphdata in enumerate_graphs(graph_file):
            index += 1
            yield (graphdata, None, index)
