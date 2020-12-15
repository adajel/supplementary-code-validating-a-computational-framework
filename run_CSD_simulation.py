import os
import sys
import glob

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# import problem, solver and plotter
from solver_BDF2 import Solver
from plotter import Plotter
from problems import *
from problem_base import ProblemBase

if __name__ == '__main__':

    # create mesh
    N = 8000                              # mesh size
    L = 0.01                              # m
    mesh = IntervalMesh(N, 0, L)          # create mesh
    boundary_point = "near(x[0], %g)" % L # point on boundary
    # time variables
    dt_value = 3.125e-3

    Tstop = 201                           # end time (s)

    # Simulation steady state with no CSD trigger
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemBase(mesh, boundary_point, t_PDE, t_ODE)
    path = 'steady_state/'
    A = [problem, path, Tstop]

    # simulation with excitatory fluxes initiating CSD
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = Problem(mesh, boundary_point, t_PDE, t_ODE)
    path = 'stim_excitatory_fluxes/'
    B = [problem, path, Tstop]

    # simulation with increased K+Cl- initiating CSD
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemStimKCl(mesh, boundary_point, t_PDE, t_ODE)
    path = 'stim_KCl/'
    C = [problem, path, Tstop]

    # simulation with pumps off - initiating CSD
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemStimPumpsOff(mesh, boundary_point, t_PDE, t_ODE)
    path = 'stim_pumpsoff/'
    D = [problem, path, Tstop]

    # problem with different gammas
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemBlockKIR(mesh, boundary_point, t_PDE, t_ODE)
    path = 'block_KIR_30/'
    E = [problem, path, Tstop]

    # problem with AQP4 deletion
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemAQP4deletion(mesh, boundary_point, t_PDE, t_ODE)
    path = 'AQP4_deletion/'
    F = [problem, path, Tstop]

    # problem with different gammas
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemGapJuncGlial(mesh, boundary_point, t_PDE, t_ODE)
    path = 'gap_junc_glial/'
    G = [problem, path, Tstop]

    # problem with different gammas
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem = ProblemNewGammas(mesh, boundary_point, t_PDE, t_ODE)
    path = 'new_gammas/'
    H = [problem, path, Tstop]

    for X in [A, B, C, D, E, F, G, H]:
        # extract problem and path
        problem = X[0]; path = X[1]; Tstop = X[2]

        # output to terminal
        print("--------------------------------")
        print(path)
        print("--------------------------------")

        # check that directory for results (data) exists, if not, create
        path_data = 'results_new/data/' + path

        if not os.path.isdir(path_data):
            os.makedirs(path_data)

        # solve system
        S = Solver(problem, dt_value, Tstop)
        S.solve_system_godenov(path_results=path_data)

        # check that directory for results (figures) exists, if yes, recreate
        path_figs = 'results_new/figures/' + path
        if not os.path.isdir(path_figs):
            os.makedirs(path_figs)

        # create plotter object for visualizing results
        P = Plotter(problem, path_data)

        # initiate wave speed and duration
        P.init_wavespeed()
        P.init_duration()

        for n in np.arange(Tstop):
            n = int(n)
            P._tmp_frames(path_figs, n)  # save plots for debugging/testing
            P.get_wavespeed(n)
            P.get_duration(n)

        P.save_wavespeed(path_figs)
        P.save_duration(path_figs)

        # save pretty summary plot of snapshot at Tstop
        P.make_spaceplot(path_figs, 60)
        P.make_timeplot(path_figs, 201)
