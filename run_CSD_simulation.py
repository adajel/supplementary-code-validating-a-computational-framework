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
    #N = 4000                              # mesh size
    L = 0.01                              # m
    mesh = IntervalMesh(N, 0, L)          # create mesh
    boundary_point = "near(x[0], %g)" % L # point on boundary
    # time variables
    dt_value = 3.125e-3
    #dt_value = 1.0e-2
    Tstop = 5                           # end time (s)
    #Tstop = 201                           # end time (s)

    # Simulation steady state with no CSD trigger
    t_PDE_A = Constant(0.0)
    t_ODE_A = Constant(0.0)
    problem_A = ProblemBase(mesh, boundary_point, t_PDE_A, t_ODE_A)
    path_A = 'steady_state/'
    A = [problem_A, path_A, Tstop]

    # simulation with excitatory fluxes initiating CSD
    t_PDE_B = Constant(0.0)
    t_ODE_B = Constant(0.0)
    problem_B = Problem(mesh, boundary_point, t_PDE_B, t_ODE_B)
    path_B = 'stim_excitatory_fluxes/'
    B = [problem_B, path_B, Tstop]

    # simulation with increased K+Cl- initiating CSD
    t_PDE_C = Constant(0.0)
    t_ODE_C = Constant(0.0)
    problem_C = ProblemStimKCl(mesh, boundary_point, t_PDE_C, t_ODE_C)
    path_C = 'stim_KCl/'
    C = [problem_C, path_C, Tstop]

    # simulation with pumps off - initiating CSD
    t_PDE_D = Constant(0.0)
    t_ODE_D = Constant(0.0)
    problem_D = ProblemStimPumpsOff(mesh, boundary_point, t_PDE_D, t_ODE_D)
    path_D = 'stim_pumpsoff/'
    D = [problem_D, path_D, Tstop]

    # problem with different gammas
    t_PDE_E = Constant(0.0)
    t_ODE_E = Constant(0.0)
    problem_E = ProblemBlockKIR70(mesh, boundary_point, t_PDE_E, t_ODE_E)
    path_E = 'block_KIR_70/'
    E = [problem_E, path_E, Tstop]

    # problem with different gammas
    t_PDE_F = Constant(0.0)
    t_ODE_F = Constant(0.0)
    problem_F = ProblemBlockKIR50(mesh, boundary_point, t_PDE_F, t_ODE_F)
    path_F = 'block_KIR_50_8000/'
    F = [problem_F, path_F, Tstop]

    # problem with AQP4 deletion
    t_PDE_G = Constant(0.0)
    t_ODE_G = Constant(0.0)
    problem_G = ProblemAQP4deletion(mesh, boundary_point, t_PDE_G, t_ODE_G)
    path_G = 'AQP4_deletion/'
    G = [problem_G, path_G, Tstop]

    # problem with different gammas
    t_PDE_H = Constant(0.0)
    t_ODE_H = Constant(0.0)
    problem_H = ProblemGapJuncGlial(mesh, boundary_point, t_PDE_H, t_ODE_H)
    path_H = 'gap_junc_glial/'
    H = [problem_H, path_H, Tstop]

    # problem with different gammas
    t_PDE_I = Constant(0.0)
    t_ODE_I = Constant(0.0)
    problem_I = ProblemNewGammas(mesh, boundary_point, t_PDE_I, t_ODE_I)
    path_I = 'new_gammas/'
    I = [problem_I, path_I, Tstop]

    #for X in [A, B, C, D, E, F, G, H]:
    for X in [F]:
        # extract problem and path
        problem = X[0]; path = X[1]; Tstop = X[2]

        # output to terminal
        print("--------------------------------")
        print(path)
        print("--------------------------------")

        # check that directory for results (data) exists, if not, create
        path_data = 'results/data/' + path

        if not os.path.isdir(path_data):
            os.makedirs(path_data)

        # solve system
        S = Solver(problem, dt_value, Tstop)
        S.solve_system_godenov(path_results=path_data)

        # check that directory for results (figures) exists, if yes, recreate
        path_figs = 'results/figures/' + path
        if not os.path.isdir(path_figs):
            os.makedirs(path_figs)

        # create plotter object for visualizing results
        P = Plotter(problem, path_data)

        """
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
        """
        P.make_spaceplot(path_figs, Tstop)
        P.make_timeplot(path_figs, Tstop)
