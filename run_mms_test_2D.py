from dolfin import *
#from vtkplotter.dolfin import *

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# set path to solver
sys.path.insert(1, '../')
from solver_BDF2 import Solver
from mms_2D import ProblemMMS

if __name__ == '__main__':

    # create directory for saving results if it does not already exist
    directory = "results/data/mms/2D/DG0"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2.txt"
    title_f2 = directory + "/convergence_table_Na_L2.txt"
    title_f3 = directory + "/convergence_table_K_L2.txt"
    title_f4 = directory + "/convergence_table_Cl_L2.txt"
    title_f5 = directory + "/convergence_table_phi_L2.txt"

    title_f6 = directory + "/convergence_table_alpha_H1.txt"
    title_f7 = directory + "/convergence_table_Na_H1.txt"
    title_f8 = directory + "/convergence_table_K_H1.txt"
    title_f9 = directory + "/convergence_table_Cl_H1.txt"
    title_f10 = directory + "/convergence_table_phi_H1.txt"

    f1 = open(title_f1, 'w+')
    f2 = open(title_f2, 'w+')
    f3 = open(title_f3, 'w+')
    f4 = open(title_f4, 'w+')
    f5 = open(title_f5, 'w+')

    f6 = open(title_f6, 'w+')
    f7 = open(title_f7, 'w+')
    f8 = open(title_f8, 'w+')
    f9 = open(title_f9, 'w+')
    f10 = open(title_f10, 'w+')

    # baseline time step
    dt_0 = 1.0e-2
    # baseline end time
    Tstop = dt_0

    # space resolutions
    resolutions = [3, 4, 5, 6]#, 7]
    # number of iterations
    i = 0

    for resolution in resolutions:
        # create mesh
        N = 2**resolution           # number of cells
        mesh = UnitSquareMesh(N, N) # mesh
        h = mesh.hmin()             # minimum diameter of cells
        boundary_point = "near(x[0], 1.0) and near(x[1], 1.0)"

        # time variables
        dt_value = dt_0/(4**i)      # time step
        t = Constant(0.0)           # time constant

        problem = ProblemMMS(mesh, boundary_point, t)
        # solve system
        S = Solver(problem, dt_value, Tstop, MMS_test=True)
        S.solve_system()

        print("-------------------------------")
        print("N", N)
        print("dt", dt_value)
        print("Tstop", Tstop)
        print("problem.t", float(problem.t))
        print("-------------------------------")

        # get sub functions
        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, \
                Cl_E, phi_N, phi_E  = S.w.split(deepcopy=True)

        # unwrap exact solutions
        alphaNe = problem.exact_solutions['alphaNe']
        NaNe = problem.exact_solutions['NaNe']
        KNe = problem.exact_solutions['KNe']
        ClNe = problem.exact_solutions['ClNe']
        NaEe = problem.exact_solutions['NaEe']
        KEe = problem.exact_solutions['KEe']
        ClEe = problem.exact_solutions['ClEe']
        phiNe = problem.exact_solutions['phiNe']
        phiEe = problem.exact_solutions['phiEe']

        # function space for exact solutions
        CG2 = FiniteElement('CG', mesh.ufl_cell(), 2) # define element
        DG0 = FiniteElement('DG', mesh.ufl_cell(), 0) # define element
        VE_CG = FunctionSpace(mesh, CG2)              # define function space
        VE_DG = FunctionSpace(mesh, DG0)              # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, VE_DG)   # Na intracellular

        Na_N_e = interpolate(NaNe, VE_CG)         # Na intracellular
        Na_E_e = interpolate(NaEe, VE_CG)         # Na extracellular

        K_N_e = interpolate(KNe, VE_CG)           # K intracellular
        K_E_e = interpolate(KEe, VE_CG)           # K extracellular

        Cl_N_e = interpolate(ClNe, VE_CG)         # Cl intracellular
        Cl_E_e = interpolate(ClEe, VE_CG)         # Cl extracellular

        phi_N_e = interpolate(phiNe, VE_CG)       # phi intracellular
        phi_E_e = interpolate(phiEe, VE_CG)       # phi extracellular

        #from vtkplotter.dolfin import plot
        #plot(phi_E_e, interactive=True)
        #plot(phi_E, interactive=True)

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=4)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=4)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=4)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=4)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=4)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=4)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=4)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=4)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=4)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=4)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=4)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=4)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=4)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=4)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=4)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=4)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=4)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=4)

        if i > 0:
            # L2 errors
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(h/h0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(h/h0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(h/h0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(h/h0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(h/h0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(h/h0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(h/h0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(h/h0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(h/h0)

            r_alphaN_H1 = np.log(alphaN_H1/alphaN_H1_0)/np.log(h/h0)
            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(h/h0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(h/h0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(h/h0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(h/h0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(h/h0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(h/h0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(h/h0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(h/h0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E(%.2f) \\\\' % (N, alphaN_L2, r_alphaN_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_L2, r_NaN_L2, NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_L2, r_KN_L2, KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_L2, r_ClN_L2, ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            phiN_L2, r_phiN_L2, phiE_L2, r_phiE_L2))

            # write to file - L2/H1 err and rate - alpha
            f6.write('%g & %.2E(%.2f) \\\\' % (N,\
                            alphaN_H1, r_alphaN_H1))
            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_H1, r_NaN_H1, NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_H1, r_KN_H1, KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_H1, r_ClN_H1, ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            phiN_H1, r_phiN_H1, phiE_H1, r_phiE_H1))

        # update prev h
        h0 = h
        # update prev L2
        alphaN_L2_0, NaN_L2_0, NaE_L2_0, \
                KN_L2_0, KE_L2_0, ClN_L2_0, ClE_L2_0,\
                phiN_L2_0,  phiE_L2_0 = alphaN_L2, \
                NaN_L2, NaE_L2, KN_L2, \
                KE_L2, ClN_L2, ClE_L2, phiN_L2, phiE_L2, \
        # update prev H1
        alphaN_H1_0, NaN_H1_0,  NaE_H1_0, \
                KN_H1_0, KE_H1_0, ClN_H1_0, ClE_H1_0,\
                phiN_H1_0, phiE_H1_0 = alphaN_H1, \
                NaN_H1, NaE_H1, KN_H1, \
                KE_H1, ClN_H1, ClE_H1, phiN_H1, phiE_H1

        # update iteration number
        i += 1

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()
