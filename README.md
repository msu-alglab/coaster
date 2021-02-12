### Requirements

* Python 3
* numpy
* scipy
* ortools (needed for finding mincost flow in the inexact flow decomposition
    heuristic)

### Running Coaster

#### FPT mode

FPT mode is the default mode for Coaster. It can be run using

```
python coaster.py [input-file]
```

#### Heuristic mode

Coaster can be run in heuristic mode to find solutions in larger graphs.

*TODO: does heuristic mode work on cyclic graphs?*

```
python coaster.py [input-file] --heuristic
```

### Submodules

The directory `ifd-package` contains code for an inexact flow solver, which is
used as part of the heuristic version of Coaster. *It contains a very similar
graph class to Coaster's graph class and should be integrated in as part of the
overall Coaster system instead of being its own submodule at some point.*

### Notes

In creating the IFD graph and in converting paths back, we detect prefix/suffix
overlaps. Both are done in an ugly way, and separately. Can we pull out into a
nice function?
