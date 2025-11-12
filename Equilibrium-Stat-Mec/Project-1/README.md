Code used for Lab 1 of the Statistical Mechanics of Complex Systems course

## Monte Carlo simulations of hard disks

How the code must be used:

First, import the main module Spheres2.py

This allows the creation of two different objects `Box` or `HardBox`.
The first one has periodic boundary conditions, and the second has impenetrable walls

To create a `Box` object you have to specify the paking fraction (phi) and the number of disks in the box (N), and it will create a square box.
Optionally you can also change the sphere diameter (sigma).

For the `HarBox`, you can either specify the paking fraction (phi), or the explicit box dimensions (Lx and Ly). Optionally, you can also include
gravity in the system (g), or change the temperature (T).

The way to run simulations is to first initialize the disks with one of two methods
`box.initialize_random_disks()` or `box.initialize_triangular_lattice()`.

And then run one of the two simulation methods
`box.run_simulation(delta, steps, save_interval=1)` Runs the simulation and saves the trajectory in a Trajectory object, returns the final positions of the disks
`box.run_chill_simulation(delta, steps)` Runs the simulation without saving the trajectory and only returns the final positions.
