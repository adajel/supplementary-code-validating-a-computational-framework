from dolfin import *

import os
import sys
import numpy as np

# set path to solver
from problems_new import *
from plotter import Plotter

if __name__ == '__main__':
    # mesh
    N = 8000                                # mesh size
    L = 0.01                                # m
    mesh = IntervalMesh(N, 0, L)            # create mesh
    boundary_point = "near(x[0], %g) % L"   # point on boundary

    # time variables
    dt_value = 3.125e-3                     # time step (s)
    Tstop = 201                             # end time (s)

    # WT problem (to compare with)
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem_WT = Problem(mesh, boundary_point, t_PDE, t_ODE)
    path_WT = 'results_new/data/stim_excitatory_fluxes/'

    # block AQP4
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem_AQP = ProblemAQP4deletion(mesh, boundary_point, t_PDE, t_ODE)
    path_AQP4 = 'AQP4_deletion/'
    A = [problem_AQP, path_AQP4]

    # alter glial gap junctions
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem_GJG = ProblemGapJuncGlial(mesh, boundary_point, t_PDE, t_ODE)
    path_GJG = 'gap_junc_glial/'
    B = [problem_GJG, path_GJG]

    # block KIR 4.1
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem_KIR = ProblemBlockKIR(mesh, boundary_point, t_PDE, t_ODE)
    path_KIR = 'block_KIR_30/'
    C = [problem_KIR, path_KIR]

    # alter membrane area to unit volume ratio
    t_PDE = Constant(0.0)
    t_ODE = Constant(0.0)
    problem_gam = ProblemNewGammas(mesh, boundary_point, t_PDE, t_ODE)
    path_gam = 'new_gammas/'
    D = [problem_gam, path_gam]

    for X in [B, C, D]:
        # create plotter object for visualizing results
        P = Plotter(problem_WT, path_WT)

        problem_2 = X[0]; path_2 = X[1]
        path_data_2 = "results_new/data/" + path_2
        P.set_mesh_and_datafile_compare(path_data_2)

        print("-----------------------------------")
        print(path_data_2)
        print("-----------------------------------")

        path_figs = "results_new/figures/" + path_2
        #P.make_spaceplot_compare(problem_2, path_figs, 60)
        #P.make_timeplot_compare(problem_2, path_figs, 75)
        P.make_timeplot_compare(problem_2, path_figs, 200)
