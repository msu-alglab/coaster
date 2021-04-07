### Requirements

* Python 3
* numpy
* scipy
* [ortools](https://developers.google.com/optimization) (needed for finding mincost flow in the inexact flow decomposition
    heuristic)

### Running Coaster

#### FPT mode

FPT mode is the default mode for Coaster. It can be run using

```
python coaster.py [input-file]
```

#### IFD Heuristic mode

Coaster can be run in IFD heuristic mode to find solutions in larger graphs using a reduction to IFD.
Note that this requires acyclic, non-nested subpath constraints, which is quite limiting.

*TODO: check exactly when heuristic mode works on cyclic graphs.*

```
python coaster.py [input-file] --ifd_heuristic
```

#### (in progress) FD heuristic mode

Coaster can also be run in FD heuristic mode, which finds solutions by first
removing any bridge edges and then using a reduction to a flow network.

```
python coaster.py [input-file] --fd_heuristic
```

### Testing

The directory `big_sc_test` contains 5,291 graphs with 2 subpath constraints
each. Only 52 of them are ANN. But can run

```
python coaster.py big_sc_test/sc0.graph --ifd_heuristic
```

and see if any errors occur.

This data file was created by running the following from the `coaster-experiments`
repository:

```
python create_sc_instances.py basic_instances/ acyclic_sc_graph_instances/ acyclic_sc_graph_instances 2 False 2 100000
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
