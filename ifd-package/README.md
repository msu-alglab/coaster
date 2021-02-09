This repository is for creating a version of the IFD heuristic solver that can
be used inside Coaster to solve a subpath constraint flow decomposition
problem.

Goal is to have something like
```
from ifd-package import InexactFlowInstance

ifd = InexactFlowInstance()
# add edges, verts from reduction process with something like:
ifd.add_vert(vert)
ifd.add_edge(edge, interval)
# solve with something like
ifd.solve()
# extract solution with something like
paths, weights = ifd.solution()
```


