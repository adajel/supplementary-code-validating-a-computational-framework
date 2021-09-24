# BDF2 time stepping scheme #

This directory contains an implementation of a numerical scheme for solving the
Mori model in the zero flow limit with two three compartments (neurons, glial
and ECS) and Na, K, Cl, Glu. Numerical scheme: BDF2 for time stepping,
ESDIRK4 for ODE time stepping (can be altered in solve_BDF2.py) and a
Strang or a Godenov splitting scheme.

### Dependencies ###

To get the environment needed (all dependencies etc.) to run the code, download
the docker container by running:

    docker run -t -v $(pwd):/home/fenics -i quay.io/fenicsproject/stable
    sudo apt-get update
    sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng

### Usage ###

The numerical experiments can be run:

    python3 run_CSD_simulation.py

### Files ###

* *run_CSD_simulation.py*  
    Run different CSD scenarios (specifies in problems.py) and plot results 

    - Output: generates time and space plots for all state variables,
              and calculates wave speed and duration.

* *solver_BDF2.py*  
    Contains class for a FEM solver for the mori model.  Numerical scheme: BDF2
    for time stepping, ESDIRK4 for ODE time stepping (can be altered in
    solve_BDF2.py) and a Strang (solve_system_strange()) or Godenov
    splitting scheme (solve_system_godenov()).

* *problem_base.py*  
    Contains class for problem base, specifying model parameters, initial
    conditions, membrane model (ODEs and/or algebraic expressions).

* *problems.py*  
    Contains class for problems, specifying triggering mechanism (excitatory
    fluxes, drop of ECS K/Cl, or turning off Na/K/ATPase) and model parameteres
    (e.g. AQP4 and KIR knockouts)

* *plotter.py*  
    Contains class for plotting.

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
