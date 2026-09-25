# Cloth Simulation

**Coming soon.**

This folder will contain our cloth simulation code, a modified version of [ARCSim](http://graphics.berkeley.edu/resources/ARCSim/). We modified its collision handling stage to record the normal contact force and the tangential relative velocity at each cloth vertex. From these, we integrate the frictional work over time to produce a wear map.

In the meantime, the wear maps used in the paper are available in [`../texture-synthesis/data/colormaps_each_fabrics/`](../texture-synthesis/data/colormaps_each_fabrics).
