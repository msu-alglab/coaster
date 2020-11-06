import random
import argparse
import os
import shutil
from toboggan.parser import read_instances

duplicates = 1


def write_tf(of, solutions, graph):
    """Write the truth file to file."""
    of.write("# graph number = {} name = {}\n".format(graph.graph_number,
                                                      graph.name))
    for weight, path in solutions:
        output_list = [weight] + path
        output = " ".join([str(x) for x in output_list]) + "\n"
        of.write(output)


def create_cf_instance(gf, tf, graph, solutions):
    for i in range(len(solutions)):
        sol = solutions[i]
        # permute
        if sol[0] > 1:
            weight = random.randrange(1, sol[0], 1)
        else:
            weight = 1
        path = sol[1].copy()
        source = path[0]
        sink = path[-1]
        middle = path[1:-1]
        random.shuffle(middle)
        new_path = [source] + middle + [sink]
        solutions.append((weight, new_path))
        graph.add_path(weight, new_path)
    graph.write(gf)
    write_tf(tf, solutions, graph)


def main(args):

    directory = args.dir
    new_directory = args.new_dir
    print("graph and truth files are in", directory)
    files = os.listdir(directory)
    graph_files = sorted([x for x in files if x.split(".")[1] == "graph"])
    truth_files = sorted([x for x in files if x.split(".")[1] == "truth"])
    print(graph_files)
    print(truth_files)
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
    os.mkdir(new_directory)
    for graph_file, truth_file in zip(graph_files, truth_files):
        gf = open(new_directory + graph_file, "w")
        tf = open(new_directory + truth_file, "w")
        print("# processing graph file", graph_file)
        graph_file = directory + graph_file
        truth_file = directory + truth_file
        for graphdata, solutions, _ in read_instances(graph_file,
                                                      truth_file):
            graph, graphname, graphnumber = graphdata
            if len(solutions) > 1 and len(solutions) <= 5:
                create_cf_instance(gf, tf, graph, solutions)
    gf.close()
    tf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("num_sc", help="number of subpaths to create",
    #                    type=str)
    parser.add_argument('dir',
                        help='A directory containing the input .graph files')
    parser.add_argument('new_dir',
                        help='A directory to output the graph and truth files')
    args = parser.parse_args()
    main(args)
