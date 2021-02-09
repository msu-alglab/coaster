Required packages:
* numpy
* scipy
* ortools (needed for finding mincost flow in the inexact flow decomposition
    heuristic)

### Submodules

The directory `ifd-package` contains code for an inexact flow solver, which is
used as part of the heuristic version of Coaster. *It contains a very similar
graph class to Coaster's graph class and should be integrated in as part of the
overall Coaster system instead of being its own submodule at some point.*
