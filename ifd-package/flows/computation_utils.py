def write_result(output_file, data):
    with open(output_file, 'a') as f:
        f.write(data)


def get_groundtruth_from_file(truth_filename):
    with open(truth_filename, 'r') as f:
      filecontents = f.read()
      graphs = filecontents.split("# graph number = ")
      graphs_dict = dict()
      for graph in graphs[1:]:
        graph_number = graph.split(" ")[0]
        graph_name = graph.split("name = ")[1].split("\n")[0]
        paths = len(graph.split("\n")) - 2
        graphs_dict[(graph_number, graph_name)] = paths
    return graphs_dict


def write_headers(output_file):
    names = get_names()
    with open(output_file, 'a') as f:
        f.write("\t".join(names) + "\n")


def get_names():
    return ['filename',
    'graphnumber',
    'confidence_level',
    'n_input',
    'm_input',
    'found_flow',
    'num_not_in_interval',
    'num_zero_lower_bounds',
    'heuristic_1_updates',
    'time_to_do_heuristic_1',
    'found_k',
    'groundtruth_k',
    'graphname',
    'splices',
    'p3_deletions',
    'find_feasible_flow_time',
    'splice_time',
    'time_to_complete_graph']


def get_true_flows_from_file(original_filename):
    """Return a dictionary of dictionaries of true flow values."""
    true_flows_dict = dict()
    index = 0
    with open(original_filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            l = lines[i]
            if l[0] == "#":
                # header line
                this_graph_dict = dict()
            elif " " not in l:
                # num verts line
                pass
            else:
                # edge line
                attributes = l.split(" ")
                vert_from = int(attributes[0])
                vert_to = int(attributes[1])
                true_flow = int(float(attributes[2]))
                this_graph_dict[(vert_from, vert_to)] = true_flow
            if i == (len(lines) -1) or lines[i+1][0] == "#":
                # end of graph
                true_flows_dict[index] = this_graph_dict
                index += 1
    return(true_flows_dict)


def count_num_true_flows_not_in_interval(graph, true_flows_dict):
    """Count the number of times the true flow not in found interval."""
    # for each edge in the graph, see if true_flows_dict[edge] is in
    # [lb, ub].
    count = 0
    for edge_info in graph.get_edge_info():
        start = edge_info[0]
        destin = edge_info[1]
        lb = edge_info[2]
        ub = edge_info[3]
        if true_flows_dict[(start, destin)] < lb:
            count += 1
        if true_flows_dict[(start, destin)] > ub:
            count += 1
    return count


def compute_weighted_jaccard(predicted_filename, groundtruth_filename, index):
    """Compute the weighted jaccard for this graph."""
    predicted_paths = []
    predicted_weights = []
    groundtruth_paths = []
    groundtruth_weights = []
    # read in predicted paths
    # there will only ever be one predicted path in the file, so can read it
    # directly.
    with open(predicted_filename, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip()
            weight = int(line.split(" ")[0])
            path = tuple(line.split(" ")[1:])
            predicted_paths.append(path)
            predicted_weights.append(weight)

    # read in groundtruth paths
    # all graphs are stored in file, so we must find the one we are interested
    # in and record only that one.
    this_graph = False
    with open(groundtruth_filename, 'r') as f:
        graph_count = 1 # index starts at 1
        line = f.readline()
        iteration = 0
        while line:
            if line[0] != "#":
                if this_graph:
                    weight = int(line.split(" ")[0])
                    path = tuple(line.split(" ")[1:])
                    groundtruth_paths.append(path)
                    groundtruth_weights.append(weight)
            # only look at the groundtruth paths for this specific graph
            if line[0] == "#":
                if graph_count == index:
                    # if this is correct graph, set a flag so that next time
                    # through the loop we start to use the information
                    this_graph = True
                else:
                    # if not the correct graph, set the flag so that
                    # we do not use the information
                    this_graph = False
                graph_count += 1
            line = f.readline().strip()

    overall_paths = list(set(groundtruth_paths + predicted_paths))

    groundtruth_vec = [0]*len(overall_paths)
    predicted_vec = [0]*len(overall_paths)

    for path, weight in zip(groundtruth_paths, groundtruth_weights):
        index = overall_paths.index(path)
        groundtruth_vec[index] = weight

    for path, weight in zip(predicted_paths, predicted_weights):
        index = overall_paths.index(path)
        predicted_vec[index] = weight

    numerator = 0
    denominator = 0

    print("Ground truth paths are:")
    print(groundtruth_paths)
    print("Groundtruth weights are:")
    print(groundtruth_weights)
    print("Predicted paths are:")
    print(predicted_paths)
    print("Predicted weights are:")
    print(predicted_weights)
    for x, y in zip(groundtruth_vec, predicted_vec):
        print(x,y)
        numerator += min(x, y)
        denominator += max(x, y)

    wj_sim = numerator/denominator

    print("Number of groundtruth paths: {}".format(len(groundtruth_paths)))
    print("Number of predicted paths: {}".format(len(predicted_paths)))

    return(wj_sim, len(groundtruth_paths), len(predicted_paths))
