We are currently on the HoreKa cluster as described in the HOREKA.md file.
The goal of this session is to create multi-GPU and ideally also multi-node
scaling tests for the astronomix simulator.

The different simulation setups are given in the pytests folder,
namely

- a hydrodynamics setup
- a magnetohydrodynamics setup
- a hydrodynamics setup with self-gravity

Astronomix implements different solver modes, here we will focus on

- the finite difference solver (FD) with native JAX backend
- the finite difference solver (FD) with PALLAS backend and different pallas_block_shape
- the finite volume solver (FV) with native JAX backend

On a single GPU (please note down which GPU your are using for each test),

- do runtime and memory usage tests for each of the simulation setups and solver modes, akin to what is already there - do not overwrite the existing data
- test the effect of varying the pallas_block_shape on the hydrodynamics setup

On multiple GPUs (for instance 4 H200s),

- test the strong scaling of all solver modes and all setups, use the insights for the ideal pallas_block_shape from before (if you think an additional test is not neccessary here) - there are already strong scaling tests on which you can orient yourself

Now to the biggest test, multi-node scaling.

- for multi-node scaling, we will focus on pure hydrodynamics with the FD solver with PALLAS backend - does anything has to be done on the code to make multi-node scaling work; the simple strong scaling test of before where we ran the same simulation on 1 vs 4 GPUs will be more difficult here (because the larger simulations now available will not run on a single GPU) - extrapolate / do weak scaling tests

There are a few things which I would like you to note:

- please log all results properly (which kind of GPU, how many, ...) and save them such that we can easily create new plots later on and adapt the style of the plots
- to see advantages in strong scaling, the problem size needs to be sufficiently large
- there are different mechanisms to save memory: write a function setting up the simulation only returning the initial state which we actually need (so that we do not keep other residues from initialization in storage), use the memory saving Runge-Kutta integrator (LSRK4), donate the state to the time_integration function, adapt the pallas_block_shape, ...
- for multi-node scaling, it would be great if we could go up to 4096^3 or at least 2048^3 cells
- please check on the GPU hours available on this account, you can use all of them but use them wisely
- when you are setting up the runner scripts keep in mind that based on the simulation time you request, you will have to wait longer or shorter in the queue
- you might need to adapt t_end of the simulations to fit the very big runs into the maximum runtimes (we would rather not get into checkpointing)

Please plan and test everything thoroughly and ask question if you have any.