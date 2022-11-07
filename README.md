A project to evaluate the usage of CUDA/OpenMP usage on Piz Daint. The code simulates 2D particle dynamics with _binning algorithm_, where a particle is assumed to collide with at maximum 6 other particles in an L-shaped neighborhood. Collision detection then happens in each cell, which is further dispatched to GPU.

Corresponding hardware configurations:

CPU: Intel® Xeon® E5-2670 @ 2.60GHz (8 cores 32GB RAM) 
GPU: NVIDIA® Tesla® K20X (Memory clock: 2.6 GHz, Memory size: 6 GB Memory I/O: 384-bit GDDR5)