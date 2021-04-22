### Requirements

* Python 3
* numpy
* scipy
* [ortools](https://developers.google.com/optimization) (needed for finding mincost flow in the inexact flow decomposition
    heuristic) TODO: remove this, since the IFD reduction is not currently
	being used.

### Running Coaster

#### FPT mode

FPT mode is the default mode for Coaster. It can be run using

```
python coaster.py [input-file]
```

#### FD heuristic mode

Coaster can also be run in FD heuristic mode, which finds solutions by first
removing any bridge edges and then using a reduction to a flow network.

```
python coaster.py [input-file] --fd_heuristic
```

#### IFD Heuristic mode (not currently in use)

Coaster can be run in IFD heuristic mode to find solutions in larger graphs using a reduction to IFD.
Note that this requires acyclic, non-nested subpath constraints, which is quite limiting.

*TODO: check exactly when ifd heuristic mode works on cyclic graphs.*

```
python coaster.py [input-file] --ifd_heuristic
```


### Testing

The file `acyclic_sc_graph_instances/len4dem1subpaths2/graphs/sc0.graph` contains
1999 graphs with 2 subpath constraints of length 4 (in the contracted graph)
each. (`len4` meaning "length 4," `dem1` meaning "demand 1," and `subpaths2`
meaning "2 subpaths.") The file
`acyclic_sc_graph_instances/len4dem1subpaths2/truth/graphs.truth` contains the corresponding ground truth paths
and weights.

Run

```
python coaster.py acyclic_sc_graph_instances/len4dem1subpaths2/graphs/sc0.graph --indices 1
```

to run the FPT algorithm on the first graph in the file. The predicted paths
and weights will be placed in the file
`acyclic_sc_graph_instances/len4dem1subpaths2/predicted_fpt/pred0.txt`.
Similarly, run
```
python coaster.py acyclic_sc_graph_instances/len4dem1subpaths2/graphs/sc0.graph --indices 1 --fd_heuristic
```
to run the same graph using the heuristic.

The file in `acyclic_sc_graph_instances/graphs/` was created by running the following from the `coaster-experiments`
repository:

```
python create_sc_instances.py basic_instances/ acyclic_sc_graph_instances/ acyclic_sc_graph_instances 2 False 4 2000
```

*TODO: use a testing module*

### Submodules

The directory `ifd-package` contains code for an inexact flow solver, which is
used as part of the heuristic version of Coaster. *It contains a very similar
graph class to Coaster's graph class and should be integrated in as part of the
overall Coaster system instead of being its own submodule at some point.*

### Notes

In creating the IFD graph and in converting paths back, we detect prefix/suffix
overlaps. Both are done in an ugly way, and separately. Can we pull out into a
nice function?
