from __future__ import print_function

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import numpy as np
from dolfin import *

import string
import os
import subprocess

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

# font size of labels
fosi = 17

# set colors
colormap = cm.viridis
mus = [1,2,3,4,5,6]
colorparams = mus
colormap = cm.viridis
normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

c2 = colormap(normalize(mus[0]))
c1 = colormap(normalize(mus[1]))
c0 = colormap(normalize(mus[2]))
c3 = colormap(normalize(mus[3]))
c4 = colormap(normalize(mus[4]))
c5 = colormap(normalize(mus[5]))

fs = 0.85
lw = 3.5

class Plotter():

    def __init__(self, problem, path_data=None):
        self.problem = problem
        N_ions = self.problem.N_ions
        N_comparts = self.problem.N_comparts
        self.N_unknows = N_comparts*(2 + N_ions) - 1

        # initialize mesh and data file
        if path_data is not None:
            self.set_mesh_and_datafile(path_data)

        return

    def set_mesh_and_datafile(self, path_data):

        # read data file
        h5_fname = path_data + 'PDE/results.h5'
        self.hdf5 = HDF5File(MPI.comm_world, h5_fname, 'r')
        # create mesh
        self.mesh = Mesh()
        self.hdf5.read(self.mesh, '/mesh', False)
        # convert coordinates from m to mm
        self.mesh.coordinates()[:] *= 1e3

        return

    def set_mesh_and_datafile_compare(self, path_data):

        # read data file
        h5_fname_compare = path_data + 'PDE/results.h5'
        self.hdf5_compare = HDF5File(MPI.comm_world, h5_fname_compare, 'r')
        # create mesh
        self.mesh_compare = Mesh()
        self.hdf5_compare.read(self.mesh_compare, '/mesh', False)
        # convert coordinates from m to mm
        self.mesh_compare.coordinates()[:] *= 1e3

        return

    def read_from_file(self, n, i, scale=1.):
        """ get snapshot of solution w[i] at time = n seconds """
        N_comparts = self.problem.N_comparts
        N_unknows = self.N_unknows

        DG0 = FiniteElement('DG', self.mesh.ufl_cell(), 0)
        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        e = [DG0]*(N_comparts - 1) + [CG1]*(N_unknows - (N_comparts - 1))

        W = FunctionSpace(self.mesh, MixedElement(e))
        u = Function(W)

        if i < (N_comparts - 1):
            V_DG0 = FunctionSpace(self.mesh, DG0)
            f = Function(V_DG0)
        else:
            V_CG1 = FunctionSpace(self.mesh, CG1)
            f = Function(V_CG1)

        # read native data file
        self.hdf5.read(u, "/solution/vector_" + str(n))

        # assign data to function
        assign(f, u.split()[i])
        f.vector()[:] = scale*f.vector().get_local()

        return f

    def read_from_file_compare(self, n, i, scale=1.):
        N_comparts = self.problem.N_comparts
        N_unknows = self.N_unknows

        DG0 = FiniteElement('DG', self.mesh_compare.ufl_cell(), 0)
        CG1 = FiniteElement('CG', self.mesh_compare.ufl_cell(), 1)
        e = [DG0]*(N_comparts - 1) + [CG1]*(N_unknows - (N_comparts - 1))

        W = FunctionSpace(self.mesh_compare, MixedElement(e))
        u = Function(W)

        if i < (N_comparts - 1):
            V_DG0 = FunctionSpace(self.mesh_compare, DG0)
            f = Function(V_DG0)
        else:
            V_CG1 = FunctionSpace(self.mesh_compare, CG1)
            f = Function(V_CG1)

        # read native data file
        self.hdf5_compare.read(u, "/solution/vector_" + str(n))

        # assign data to function
        assign(f, u.split()[i])
        f.vector()[:] = scale*f.vector().get_local()

        return f

    def project_to_function_space(self, u):
        """ project u onto function space """

        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        V = FunctionSpace(self.mesh, CG1)
        f = project(u, V)

        return f

    def make_amplitude(self, path_figs, n):
        """ check contribution from amplitude at given time n """

        # read solutions from file
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        #Na_N = self.read_from_file(n, 2)
        #Na_G = self.read_from_file(n, 3)
        #Na_E = self.read_from_file(n, 4)
        #K_N = self.read_from_file(n, 5)
        #K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        #Cl_N = self.read_from_file(n, 8)
        #Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        #Glu_N = self.read_from_file(n, 11)
        #Glu_G = self.read_from_file(n, 12)
        Glu_E = self.read_from_file(n, 13)
        # concert to mV
        phi_N = self.read_from_file(n, 14, scale=1.0e3)
        phi_G = self.read_from_file(n, 15, scale=1.0e3)
        phi_E = self.read_from_file(n, 16, scale=1.0e3)

        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        K_E_init = float(self.problem.K_E_init)
        Cl_E_init = float(self.problem.Cl_E_init)
        Glu_E_init = float(self.problem.Glu_E_init)
        phi_N_init = float(self.problem.phi_N_init)*1.0e3
        phi_G_init = float(self.problem.phi_G_init)*1.0e3
        phi_E_init = float(self.problem.phi_E_init)*1.0e3

        phi_NE_init = phi_N_init - phi_E_init
        phi_GE_init = phi_G_init - phi_E_init

        phi_NE = self.project_to_function_space(phi_N - phi_E)
        phi_GE = self.project_to_function_space(phi_G - phi_E)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # calculate amplitudes

        # ECS concentrations
        self.amplitude_K_E = max(abs(K_E.vector().get_local() - K_E_init))
        self.amplitude_Glu_E = max(abs(Glu_E.vector().get_local() - Glu_E_init))
        self.amplitude_Cl_E = max(abs(Cl_E.vector().get_local() - Cl_E_init))

        # potentials
        self.amplitude_phi_NE = max(abs(phi_NE.vector().get_local() - phi_NE_init))
        self.amplitude_phi_GE = max(abs(phi_GE.vector().get_local() - phi_GE_init))
        self.amplitude_phi_E = max(abs(phi_E.vector().get_local() - phi_E_init))

        # volume fractions
        self.amplitude_alpha_N = max(abs(alpha_N_diff.vector().get_local()))
        self.amplitude_alpha_G = max(abs(alpha_G_diff.vector().get_local()))
        self.amplitude_alpha_E = max(abs(alpha_E_diff.vector().get_local()))

        # write amplitudes to file
        title_f = path_figs + "amplitude.txt"
        f = open(title_f, 'w+')

        f.write('ECS concentrations:')
        f.write('\n')
        f.write('amplitude K_E %g' % self.amplitude_K_E)
        f.write('\n')
        f.write('amplitude Glu_E %g' % self.amplitude_Glu_E)
        f.write('\n')
        f.write('amplitude Cl_E %g' % self.amplitude_Cl_E)
        f.write('\n')
        f.write('\n')

        f.write('Potentials:')
        f.write('\n')
        f.write('amplitude phi_NE %g' % self.amplitude_phi_NE)
        f.write('\n')
        f.write('amplitude phi_GE %g' % self.amplitude_phi_GE)
        f.write('\n')
        f.write('amplitude phi_E %g' % self.amplitude_phi_E)
        f.write('\n')
        f.write('\n')

        f.write('Volume fractions:')
        f.write('\n')
        f.write('amplitude alpha_N %g' % self.amplitude_alpha_N)
        f.write('\n')
        f.write('amplitude alpha_G %g' % self.amplitude_alpha_G)
        f.write('\n')
        f.write('amplitude alpha_E %g' % self.amplitude_alpha_E)
        f.write('\n')

        f.close()

        return

    def init_duration(self):
        # initiate calculation of duration (s)

        # ECS concentrations
        self.duration_K_E = 0
        self.duration_Glu_E = 0
        self.duration_Cl_E = 0

        # potentials
        self.duration_phi_NE = 0
        self.duration_phi_GE = 0
        self.duration_phi_E = 0

        # volume fractions
        self.duration_alpha_N = 0
        self.duration_alpha_G = 0
        self.duration_alpha_E = 0

        return

    def get_duration(self, n):
        """ check contribution from duration at given time n """

        # read solutions from file
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        #Na_N = self.read_from_file(n, 2)
        #Na_G = self.read_from_file(n, 3)
        #Na_E = self.read_from_file(n, 4)
        #K_N = self.read_from_file(n, 5)
        #K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        #Cl_N = self.read_from_file(n, 8)
        #Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        #Glu_N = self.read_from_file(n, 11)
        #Glu_G = self.read_from_file(n, 12)
        Glu_E = self.read_from_file(n, 13)
        # concert to mV
        phi_N = self.read_from_file(n, 14, scale=1.0e3)
        phi_G = self.read_from_file(n, 15, scale=1.0e3)
        phi_E = self.read_from_file(n, 16, scale=1.0e3)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)
        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        phi_NE = self.project_to_function_space(phi_N - phi_E)
        phi_GE = self.project_to_function_space(phi_G - phi_E)

        # point at which to evaluate solution
        point = 1.0

        # evaluate ECS concentrations at point
        K_E_x = K_E(point)
        Glu_E_x = Glu_E(point)
        Cl_E_x = Cl_E(point)

        # evaluate potentials at point
        phi_NE_x = phi_NE(point)
        phi_GE_x = phi_GE(point)
        phi_E_x = phi_E(point)

        # evaluate volume fractions at point
        alpha_N_x = alpha_N_diff(point)
        alpha_G_x = alpha_G_diff(point)
        alpha_E_x = alpha_E_diff(point)

        # add one second to duration if wave is present (i.e u > u_thres):
        # if value is greater/lesser than threshold for ECS concentrations (mM)
        if K_E_x > 8: self.duration_K_E += 1            # init 4 mM
        if Glu_E_x > 0.02: self.duration_Glu_E += 1     # init 0.01 mM
        if Cl_E_x < 111: self.duration_Cl_E += 1        # init 113 mM

        # add one second to duration if wave is present (i.e k > k_thres):
        # if value is greater/lesser than threshold for potentials (mV)
        if phi_NE_x > -66: self.duration_phi_NE += 1    # init -71 mV
        if phi_GE_x > -77: self.duration_phi_GE += 1    # init -82 mV
        #if phi_E_x < -0.05: self.duration_phi_E += 1    # init   0 mV
        if phi_E_x < -0.5: self.duration_phi_E += 1    # init   0 mV

        # add one second to duration if wave is present (i.e k > k_thres):
        # if value is greater/lesser than threshold for volume fractions (%)
        if alpha_N_x > 0.5: self.duration_alpha_N += 1
        if alpha_G_x > 0.5: self.duration_alpha_G += 1
        if alpha_E_x < -0.5: self.duration_alpha_E += 1

        return

    def save_duration(self, res_path):

        # ECS concentrations
        duration_K_E = self.duration_K_E
        duration_Glu_E = self.duration_Glu_E
        duration_Cl_E = self.duration_Cl_E

        # potentials
        duration_phi_NE = self.duration_phi_NE
        duration_phi_GE = self.duration_phi_GE
        duration_phi_E = self.duration_phi_E

        # volume fractions
        duration_alpha_N = self.duration_alpha_N
        duration_alpha_G = self.duration_alpha_G
        duration_alpha_E = self.duration_alpha_E

        # write durations to file
        title_f = res_path + "duration.txt"
        f = open(title_f, 'w+')

        f.write('ECS concentrations:')
        f.write('\n')
        f.write('duration K_E %g ' % self.duration_K_E)
        f.write('\n')
        f.write('duration Glu_E %g ' % self.duration_Glu_E)
        f.write('\n')
        f.write('duration Cl_E %g \n' % self.duration_Cl_E)
        f.write('\n')
        f.write('\n')

        f.write('Potentials:')
        f.write('\n')
        f.write('duration phi_NE %g ' % self.duration_phi_NE)
        f.write('\n')
        f.write('duration phi_GE %g ' % self.duration_phi_GE)
        f.write('\n')
        f.write('duration phi_E %g \n ' % self.duration_phi_E)
        f.write('\n')
        f.write('\n')

        f.write('Volume fractions:')
        f.write('\n')
        f.write('duration alpha_N %g ' % self.duration_alpha_N)
        f.write('\n')
        f.write('duration alpha_G %g ' % self.duration_alpha_G)
        f.write('\n')
        f.write('duration alpha_E %g  \n' % self.duration_alpha_E)
        f.write('\n')

        f.close()

        return

    def init_wavespeed(self):
        """ initiate calculation of wave speed """
        # for saving pairs of space and time coordinates when wave passes
        self.wavefront_space_time = []
        # get coordinates (mm)
        self.coordinates = self.mesh.coordinates()
        return

    def get_wavespeed(self, n):
        """ save wave speed at given time n """
        # get neuron potential
        phi_N = self.read_from_file(n, 14, scale=1.0e3)

        # get values of phi N
        phi_N_values = phi_N.compute_vertex_values()

        # get max value of potential (phi) at time n
        index_max = max(range(len(phi_N_values)), key=phi_N_values.__getitem__)

        v_max = phi_N_values[index_max]          # max value of phi N (mV)
        p_max = self.coordinates[index_max][0]   # point of max value (mm)
        t_max = n                                # time for max value (s)

        # check that wave has passed
        # (assumption: wave has passed if neuron potential phi  > -20 mV)
        if v_max > -20:
            # save point in space and time of wave front
            # (assumption: wave front is where phi_N has greatest value)
            self.wavefront_space_time.append([p_max, t_max])
        return

    def save_wavespeed(self, res_path):
        p_max_prev = 0  # previous value of p_max
        t_max_prev = 0  # previous value of t_max
        speeds = []     # for saving speeds

        for i, pair in enumerate(self.wavefront_space_time):
            # get values
            (p_max, t_max) = pair

            if i > 0:
                if p_max_prev == p_max:
                    # wave front has not moved to next point -> continue
                    continue
                else:
                    # wave front has moved to next point -> calculate speed
                    # distance between wave front (mm)
                    dx = p_max - p_max_prev
                    # time between wave front (convert from second to min)
                    dt = (t_max - t_max_prev)/60.0

                    # calculate and append speed
                    speeds.append(dx/dt)
                    # update p_max_prev and t_max_prev
                    p_max_prev = p_max
                    t_max_prev = t_max

        # remove effects from initiation and termination of wave
        speeds_mid = speeds[20:-20]
        # plot speeds
        if (len(speeds_mid) > 0):
            # calculate average speed
            avg = sum(speeds_mid)/len(speeds_mid)
            # plot speeds
            plt.figure()
            plt.plot(speeds_mid, 'ro')
            plt.ylabel(r'mm/min')
            plt.xlabel('intervals')
            plt.title('Wave speed, avg = %.2f mm/min' % avg)
            # save figure
            fname_res = res_path + 'wavespeed.png'
            plt.savefig(fname_res)
            plt.close()

        return speeds_mid

    def make_timeplot(self, path_figs, Tstop):
        """ plot variables at t=Tstop """

        # point at which to calculate duration
        point = 2.0

        # list of function values at point
        alpha_Ns = []; alpha_Gs = []; alpha_Es = []
        Na_Ns = []; K_Ns = []; Cl_Ns = []; Glu_Ns = []
        Na_Gs = []; K_Gs = []; Cl_Gs = []; Glu_Gs = []
        Na_Es = []; K_Es = []; Cl_Es = []; Glu_Es = []
        phi_Ns = []; phi_Gs = []; phi_Es = []

        for n in range(Tstop):

            # get and append data PDEs
            alpha_N = self.read_from_file(n, 0)
            alpha_G = self.read_from_file(n, 1)
            Na_N = self.read_from_file(n, 2)
            Na_G = self.read_from_file(n, 3)
            Na_E = self.read_from_file(n, 4)
            K_N = self.read_from_file(n, 5)
            K_G = self.read_from_file(n, 6)
            K_E = self.read_from_file(n, 7)
            Cl_N = self.read_from_file(n, 8)
            Cl_G = self.read_from_file(n, 9)
            Cl_E = self.read_from_file(n, 10)
            Glu_N = self.read_from_file(n, 11)
            Glu_G = self.read_from_file(n, 12)
            Glu_E = self.read_from_file(n, 13)
            # concert to mV
            phi_N = self.read_from_file(n, 14, scale=1.0e3)
            phi_G = self.read_from_file(n, 15, scale=1.0e3)
            phi_E = self.read_from_file(n, 16, scale=1.0e3)

            # calculate extracellular volume fraction
            u_alpha_E = 1.0 - alpha_N - alpha_G
            alpha_E = self.project_to_function_space(u_alpha_E)
            alpha_N_init = float(self.problem.alpha_N_init)
            alpha_G_init = float(self.problem.alpha_G_init)
            alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

            # calculate charge in volume fractions (alpha) in %
            u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
            u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
            u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
            alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
            alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
            alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

            alpha_Ns.append(alpha_N_diff(point))
            alpha_Gs.append(alpha_G_diff(point))
            alpha_Es.append(alpha_E_diff(point))
            Na_Ns.append(Na_N(point))
            K_Ns.append(K_N(point))
            Cl_Ns.append(Cl_N(point))
            Na_Gs.append(Na_G(point))
            K_Gs.append(K_G(point))
            Cl_Gs.append(Cl_G(point))
            Na_Es.append(Na_E(point))
            K_Es.append(K_E(point))
            Cl_Es.append(Cl_E(point))
            Glu_Ns.append(Glu_N(point))
            Glu_Gs.append(Glu_G(point))
            Glu_Es.append(Glu_E(point))
            phi_Ns.append(phi_N(point))
            phi_Gs.append(phi_G(point))
            phi_Es.append(phi_E(point))

        ######################################################################
        print("-------------------")
        print("-------------------")
        print("ION CONCENTRATIONS")
        print("-------------------")
        print("Neuron")
        print("Na 0", Na_Ns[0])
        print("Na Tstop", Na_Ns[-1])
        print("K 0", K_Ns[0])
        print("K Tstop", K_Ns[-1])
        print("Cl 0", Cl_Ns[0])
        print("Cl Tstop", Cl_Ns[-1])
        print("-------------------")
        print("Glial")
        print("Na 0", Na_Gs[0])
        print("Na Tstop", Na_Gs[-1])
        print("K 0", K_Gs[0])
        print("K Tstop", K_Gs[-1])
        print("Cl 0", Cl_Gs[0])
        print("Cl Tstop", Cl_Gs[-1])
        print("-------------------")
        print("ECS")
        print("Na 0", Na_Es[0])
        print("Na Tstop", Na_Es[-1])
        print("K 0", K_Es[0])
        print("K Tstop", K_Es[-1])
        print("Cl 0", Cl_Es[0])
        print("Cl Tstop", Cl_Es[-1])
        print("-------------------")
        print("-------------------")
        print("POTENTIALS")
        print("-------------------")
        print("N 0", phi_Ns[0])
        print("N Tstop", phi_Ns[-1])
        print("G 0", phi_Gs[0])
        print("G Tstop", phi_Gs[-1])
        print("ECS 0", phi_Es[0])
        print("ECS Tstop", phi_Es[-1])
        print("-------------------")
        print("-------------------")
        print("VOLUME FRACTIONS")
        print("-------------------")
        print("N 0", alpha_Ns[0])
        print("N Tstop", alpha_Ns[-1])
        print("G 0", alpha_Gs[0])
        print("G Tstop", alpha_Gs[-1])
        print("ECS 0", alpha_Es[0])
        print("ECS Tstop", alpha_Es[-1])
        print("-------------------")

        # range of x values
        xlim = [0.0, Tstop]

        # create plot
        fig = plt.figure(figsize=(19.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,4,1, xlim=xlim, ylim=[0.0, 1.5])
        plt.ylabel(r'[Glu$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([0.25, 0.5, 0.75, 1, 1.25])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(Glu_Es, label=r'Glu', linewidth=lw)

        ax2 = fig.add_subplot(1,4,2, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'[k$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([20, 40, 60, 80, 100, 120, 140])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(Na_Es, color=c0, label=r'Na$^+$', linewidth=lw)
        plt.plot(K_Es, color=c1,  label=r'K$^+$',linewidth=lw)
        plt.plot(Cl_Es, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax3 = fig.add_subplot(1,4,3, xlim=xlim, ylim=[-100, 20])
        plt.ylabel(r'$\phi$ (mV)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([-90, -70, -50, -30, -10, 10])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(phi_Es, color=c3, label=r'ECS', linewidth=lw)
        plt.plot(phi_Ns, color=c4, label=r'neuron',linewidth=lw)
        plt.plot(phi_Gs, color=c5, label=r'glial', linewidth=lw)

        ax4 = fig.add_subplot(1,4,4, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'$\Delta \alpha $ (\%)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([-40, -30, -20, -10, 0, 10])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(alpha_Es, color=c3, linewidth=lw)
        plt.plot(alpha_Ns, color=c4, linewidth=lw)
        plt.plot(alpha_Gs, color=c5, linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.35, left=0.1)

        # add legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.9))

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{E}', r'\textbf{F}', r'\textbf{G}', r'\textbf{H}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'timeplot_summary_200'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()

        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        ######################################################################
        # range of x values
        xlim = [0.0, Tstop]

        # create plot
        fig = plt.figure(figsize=(18.0*fs, 10*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(2,4,1, xlim=xlim, ylim=[0.0, 1.5])
        plt.title(r'ECS glutamate')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.plot(Glu_Es, color=c0, linewidth=lw)

        ax2 = fig.add_subplot(2,4,2, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'ECS ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plt.plot(Na_Es, color=c0, label=r'Na$^+$', linewidth=lw)
        plt.plot(K_Es, color=c1,  label=r'K$^+$',linewidth=lw)
        plt.plot(Cl_Es, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax3 = fig.add_subplot(2,4,3, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'neuron ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plt.plot(Na_Ns, color=c0, label=r'Na$^+$', linewidth=lw)
        plt.plot(K_Ns, color=c1,  label=r'K$^+$',linewidth=lw)
        plt.plot(Cl_Ns, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax4 = fig.add_subplot(2,4,4, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'glial ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plt.plot(Na_Gs, color=c0, label=r'Na$^+$', linewidth=lw)
        plt.plot(K_Gs, color=c1,  label=r'K$^+$',linewidth=lw)
        plt.plot(Cl_Gs, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax5 = fig.add_subplot(2,4,5, xlim=xlim, ylim=[-100, 20])
        plt.title(r'potentials')
        plt.ylabel(r'mV')
        plt.xlabel(r's')
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plt.plot(phi_Ns, color=c4, linewidth=lw)
        plt.plot(phi_Gs, color=c5, linewidth=lw)
        plt.plot(phi_Es, color=c3, linewidth=lw)

        phi_NEs = [phi_N - phi_E for phi_N, phi_E in zip(phi_Ns, phi_Es)]
        phi_GEs = [phi_G - phi_E for phi_G, phi_E in zip(phi_Gs, phi_Es)]

        ax6 = fig.add_subplot(2,4,6, xlim=xlim, ylim=[-100, 20])
        plt.title(r'membrane potentials')
        plt.ylabel(r'mV')
        plt.xlabel(r's')
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plt.plot(phi_NEs, color=c4, linewidth=lw)
        plt.plot(phi_GEs, color=c5, linewidth=lw)

        ax7 = fig.add_subplot(2,4,7, xlim=xlim, ylim=[-50, 20])
        plt.title(r'\% change volume fractions')
        plt.ylabel(r'\%')
        plt.xlabel(r's')
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plt.plot(alpha_Ns, color=c4, label=r'neuron', linewidth=lw)
        plt.plot(alpha_Gs, color=c5, label=r'glial', linewidth=lw)
        plt.plot(alpha_Es, color=c3, label=r'ECS', linewidth=lw)

        # add legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.85))

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.25)

        # define filename
        fname_res = path_figs + 'timeplot.png'
        # save figure to file
        fig.savefig(fname_res, bbox_inches='tight')
        plt.close()

        ######################################################################
        # set bar width
        barWidth = 0.55

        # create plot Glutamate
        fig = plt.figure(figsize=(14.5*fs, 10*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,3,1)
        types = [r'original', r'AQP4$^{-}$', r'KIR$^{-}$', r'gap junc', r'new $\gamma$s']
        spees = [7.43, 7.43, 7.49, 7.88, 5.4]
        plt.bar(types, spees, width=barWidth, edgecolor='black', hatch='//')
        plt.title(r"CSD wave speed propagation")
        plt.xticks([r for r in range(len(spees))], types, rotation=90)
        plt.ylim([5, 8])
        plt.yticks([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
        plt.ylabel(r"mm/min")

        ax2 = fig.add_subplot(2,3,2, xlim=xlim, ylim=[0, 150])
        plt.title(r'neuron ion concentrations')
        plt.ylabel(r'mM')
        plt.plot(Na_Ns, color=c0, label=r'Na$^{+}$', linewidth=lw)
        plt.plot(K_Ns,  color=c1, label=r'K$^{+}$', linewidth=lw)
        plt.plot(Cl_Ns, color=c2, label=r'Cl$^{-}$', linewidth=lw)
        plt.xlabel(r's')

        ax3 = fig.add_subplot(2,3,3, xlim=xlim, ylim=[0, 150])
        plt.title(r'glial ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.plot(Na_Gs, color=c0, linewidth=lw)
        plt.plot(K_Gs,  color=c1, linewidth=lw)
        plt.plot(Cl_Gs, color=c2, linewidth=lw)

        ax4 = fig.add_subplot(2,3,5, xlim=xlim, ylim=[8, 11])
        plt.title(r'neuron glutamate')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.plot(Glu_Ns, color=c4, label=r'neuron', linewidth=lw)

        ax5 = fig.add_subplot(2,3,6, xlim=xlim, ylim=[0, 1.5])
        plt.title(r'glial and ECS glutamate')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.plot(Glu_Es, color=c3, label=r'ECS', linewidth=lw)
        plt.plot(Glu_Gs, color=c5, label=r'glial',  linewidth=lw)

        # add legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.85))

        ax.axis('off')
        plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.065)
        plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.065)

        # add numbering for the subplots (A, B, C etc)
        ax1.text(-0.12, 1.025, r'\textbf{A}', transform=ax1.transAxes)
        letters = [ r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', r'\textbf{E}']
        for num, ax in enumerate([ax2, ax3, ax4, ax5]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # define filename
        fname_res = path_figs + 'timeplot_wavespeed.png'
        # save figure to file
        fig.savefig(fname_res, bbox_inches='tight')
        plt.close()

        return

    def make_spaceplot(self, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # get data
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        Na_N = self.read_from_file(n, 2)
        Na_G = self.read_from_file(n, 3)
        Na_E = self.read_from_file(n, 4)
        K_N = self.read_from_file(n, 5)
        K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        Cl_N = self.read_from_file(n, 8)
        Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        Glu_N = self.read_from_file(n, 11)
        Glu_G = self.read_from_file(n, 12)
        Glu_E = self.read_from_file(n, 13)
        # concert to mV
        phi_N = self.read_from_file(n, 14, scale=1.0e3)
        phi_G = self.read_from_file(n, 15, scale=1.0e3)
        phi_E = self.read_from_file(n, 16, scale=1.0e3)

        phi_NE = self.project_to_function_space(phi_N - phi_E)
        phi_GE = self.project_to_function_space(phi_G - phi_E)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)

        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init
        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        print("phi_N", max(phi_N.vector().get_local()))
        print("phi_G", max(phi_G.vector().get_local()))

        dx = 0.01/8000.*1000
        g_vec = phi_N.compute_vertex_values()
        index_max = max(range(len(g_vec)), key=g_vec.__getitem__)
        point_of_wf = dx*index_max
        print("point_of_wf", point_of_wf)

        # range of x values
        xlim = [0.0, 10.0]

        # create plot
        fig = plt.figure(figsize=(19.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,4,1, xlim=xlim, ylim=[0, 1.5])
        plt.ylabel(r'[Glu$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25])
        plot(Glu_E, label=r'Glu', linewidth=lw)

        ax2 = fig.add_subplot(1,4,2, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'[k$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([20, 40, 60, 80, 100, 120, 140])
        plot(Na_E, color=c0, label=r'Na$^+$', linewidth=lw)
        plot(K_E, color=c1, label=r'K$^+$',linewidth=lw)
        plot(Cl_E, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax3 = fig.add_subplot(1,4,3, xlim=xlim, ylim=[-100, 20])
        plt.ylabel(r'$\phi$ (mV)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_E, color=c3, label=r'ECS', linewidth=lw)
        plot(phi_N, color=c4, label=r'neuron', linewidth=lw)
        plot(phi_G, color=c5, label=r'glial', linewidth=lw)

        ax4 = fig.add_subplot(1,4,4, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'$\Delta \alpha $ (\%)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plot(alpha_E_diff, color=c3, linewidth=lw)
        plot(alpha_N_diff, color=c4, linewidth=lw)
        plot(alpha_G_diff, color=c5, linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.35, left=0.1)

        # add numbering for the subplots (A, B, C etc)
        if path_figs == 'results/figures/stim_KCl/':
            letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
            plt.figlegend(bbox_to_anchor=(1.0, 0.9)) # add legend
        elif path_figs == 'results/figures/stim_pumpsoff/':
            letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
            plt.figlegend(bbox_to_anchor=(1.0, 0.9)) # add legend
        else:
            letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
            plt.figlegend(bbox_to_anchor=(1.0, 0.9)) # add legend

        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'spaceplot_summary'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        # create plot
        fig = plt.figure(figsize=(14.5*fs, 5*fs))
        ax = plt.gca()
        # subplot number 1 - extracellular concentrations
        ax1 = fig.add_subplot(1,3,1, xlim=xlim, ylim=[0, 1.5])
        plt.title(r'ECS glutamate')
        plt.ylabel(r'mM')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(Glu_E, linewidth=lw)
        # subplot number 2 - potentials
        ax2 = fig.add_subplot(1,3,2, xlim=xlim, ylim=[8, 11])
        plt.title(r'neuron glutamate concentration')
        plt.ylabel(r'mV')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(Glu_N, linewidth=lw)
        # subplot number 3 - volume fractions
        ax3 = fig.add_subplot(1,3,3, xlim=xlim, ylim=[0, 1.5])
        plt.title(r'glial glutamate concentration')
        plt.ylabel(r'\%')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(Glu_G, linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.tight_layout()
        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']
        for num, ax in enumerate([ax1, ax2, ax3]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'spaceplot_glu'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        # range of x values
        xlim = [1.0, 10.0]

        # create plot
        fig = plt.figure(figsize=(19.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,4,1, xlim=xlim, ylim=[0, 15])
        plt.ylabel(r'[Glu$]$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(Glu_N, color=c4, linewidth=lw)
        plot(Glu_G, color=c5, linewidth=lw)

        ax2 = fig.add_subplot(1,4,2, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'[k$]_n$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_N, color=c0, linewidth=lw)
        plot(K_N, color=c1, linewidth=lw)
        plot(Cl_N, color=c2, linewidth=lw)

        ax3 = fig.add_subplot(1,4,3, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'[k$]_g$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_G, color=c0, label=r'Na$^+$', linewidth=lw)
        plot(K_G, color=c1, label=r'K$^+$',linewidth=lw)
        plot(Cl_G, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax4 = fig.add_subplot(1,4,4, xlim=xlim, ylim=[-100, 20])
        plt.ylabel(r'$\phi_M$ (mV)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_NE, color=c4, label=r'neuron', linewidth=lw)
        plot(phi_GE, color=c5, label=r'glial', linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.35, left=0.1)

        # add numbering for the subplots (A, B, C etc)
        if path_figs == 'results/figures/stim_KCl/':
            letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
            plt.figlegend(bbox_to_anchor=(1.0, 0.9)) # add legend
        elif path_figs == 'results/figures/stim_pumpsoff/':
            letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
            plt.figlegend(bbox_to_anchor=(1.0, 0.9)) # add legend
        else:
            letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
            plt.figlegend(bbox_to_anchor=(1.0, 0.9)) # add legend

        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'spaceplot_summary_ICS'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return

    def make_spaceplot_compare(self, problem_2, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # get data 1
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        Na_E = self.read_from_file(n, 4)
        K_E = self.read_from_file(n, 7)
        Cl_E = self.read_from_file(n, 10)
        Glu_N = self.read_from_file(n, 11)
        Glu_G = self.read_from_file(n, 12)
        Glu_E = self.read_from_file(n, 13)
        # concert to mV
        phi_N = self.read_from_file(n, 14, scale=1.0e3)
        phi_G = self.read_from_file(n, 15, scale=1.0e3)
        phi_E = self.read_from_file(n, 16, scale=1.0e3)

        # get data 2
        alpha_N_2 = self.read_from_file_compare(n, 0)
        alpha_G_2 = self.read_from_file_compare(n, 1)
        Na_E_2 = self.read_from_file_compare(n, 4)
        K_E_2 = self.read_from_file_compare(n, 7)
        Cl_E_2 = self.read_from_file_compare(n, 10)
        Glu_N_2 = self.read_from_file_compare(n, 11)
        Glu_G_2 = self.read_from_file_compare(n, 12)
        Glu_E_2 = self.read_from_file_compare(n, 13)
        # concert to mV
        phi_N_2 = self.read_from_file_compare(n, 14, scale=1.0e3)
        phi_G_2 = self.read_from_file_compare(n, 15, scale=1.0e3)
        phi_E_2 = self.read_from_file_compare(n, 16, scale=1.0e3)

        # calculate extracellular volume fraction 1
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)
        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init
        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # calculate extracellular volume fraction 2
        u_alpha_E_2 = 1.0 - alpha_N_2 - alpha_G_2
        alpha_E_2 = self.project_to_function_space(u_alpha_E_2)
        alpha_N_init_2 = float(problem_2.alpha_N_init)
        alpha_G_init_2 = float(problem_2.alpha_G_init)
        alpha_E_init_2 = 1.0 - alpha_N_init_2 - alpha_G_init_2
        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff_2 = (alpha_N_2 - alpha_N_init_2)/alpha_N_init_2*100
        u_alpha_G_diff_2 = (alpha_G_2 - alpha_G_init_2)/alpha_G_init_2*100
        u_alpha_E_diff_2 = (alpha_E_2 - alpha_E_init_2)/alpha_E_init_2*100
        alpha_N_diff_2 = self.project_to_function_space(u_alpha_N_diff_2)
        alpha_G_diff_2 = self.project_to_function_space(u_alpha_G_diff_2)
        alpha_E_diff_2 = self.project_to_function_space(u_alpha_E_diff_2)

        # plotting parameters
        xlim = [0.0, 10.0] # range of x values

        # create plot
        fig = plt.figure(figsize=(19.5*fs, 5*fs))
        ax = plt.gca()

        # subplot number 1 - extracellular glutamate
        ax1 = fig.add_subplot(1,4,1, xlim=xlim, ylim=[0.0, 1.5])
        plt.ylabel(r'[Glu$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25])
        plot(Glu_E, color='#1f77b4', label=r'Glu', linewidth=lw)
        plot(Glu_E_2, color='#1f77b4', linestyle='dashed', linewidth=lw)
        # subplot number 1 - extracellular concentrations
        ax2 = fig.add_subplot(1,4,2, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'[k$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([20, 40, 60, 80, 100, 120, 140])
        plot(Na_E, color=c0, label=r'Na$^+$', linewidth=lw)
        plot(K_E, color=c1,  label=r'K$^+$', linewidth=lw)
        plot(Cl_E, color=c2, label=r'Cl$^-$', linewidth=lw)
        plot(Na_E_2, color=c0, linestyle='dashed', linewidth=lw)
        plot(K_E_2, color=c1, linestyle='dashed', linewidth=lw)
        plot(Cl_E_2, color=c2, linestyle='dashed', linewidth=lw)

        # subplot number 2 - potentials
        ax3 = fig.add_subplot(1,4,3, xlim=xlim, ylim=[-100, 20])
        plt.ylabel(r'$\phi$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_E, color=c3, label=r'ECS', linewidth=lw)
        plot(phi_N, color=c4, label=r'neuron', linewidth=lw)
        plot(phi_G, color=c5, label=r'glial', linewidth=lw)
        plot(phi_E_2, color=c3, linestyle='dashed', linewidth=lw)
        plot(phi_N_2, color=c4, linestyle='dashed', linewidth=lw)
        plot(phi_G_2, color=c5, linestyle='dashed', linewidth=lw)
        # subplot number 3 - volume fractions
        ax4 = fig.add_subplot(1,4,4, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'$\Delta \alpha $ (\%)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(alpha_E_diff, color=c3, linewidth=lw)
        plot(alpha_N_diff, color=c4, linewidth=lw)
        plot(alpha_G_diff, color=c5, linewidth=lw)
        plot(alpha_E_diff_2, color=c3, linestyle='dashed', linewidth=lw)
        plot(alpha_N_diff_2, color=c4, linestyle='dashed', linewidth=lw)
        plot(alpha_G_diff_2, color=c5, linestyle='dashed', linewidth=lw)

        # make legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.9))

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.35, left=0.1)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'compare_with_default_space'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')
        return

    def make_timeplot_compare(self, problem_2, path_figs, Tstop):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # point at which to calculate duration
        point = 2.0

        # list of function values at point - native data
        alpha_Ns = []; alpha_Gs = []; alpha_Es = []
        Na_Es = []; K_Es = []; Cl_Es = []
        Glu_Ns = []; Glu_Gs = []; Glu_Es = []
        phi_Ns = []; phi_Gs = []; phi_Es = []
        # list of function values at point - compare to native data
        alpha_Ns_2 = []; alpha_Gs_2 = []; alpha_Es_2 = []
        Na_Es_2 = []; K_Es_2 = []; Cl_Es_2 = []
        Glu_Ns_2 = []; Glu_Gs_2 = []; Glu_Es_2 = []
        phi_Ns_2 = []; phi_Gs_2 = []; phi_Es_2 = []

        for n in range(Tstop):
            # get data PDEs
            alpha_N = self.read_from_file(n, 0)
            alpha_G = self.read_from_file(n, 1)
            Na_E = self.read_from_file(n, 4)
            K_E = self.read_from_file(n, 7)
            Cl_E = self.read_from_file(n, 10)
            Glu_N = self.read_from_file(n, 11)
            Glu_G = self.read_from_file(n, 12)
            Glu_E = self.read_from_file(n, 13)
            # concert to mV
            phi_N = self.read_from_file(n, 14, scale=1.0e3)
            phi_G = self.read_from_file(n, 15, scale=1.0e3)
            phi_E = self.read_from_file(n, 16, scale=1.0e3)

            # calculate extracellular volume fraction 1
            u_alpha_E = 1.0 - alpha_N - alpha_G
            alpha_E = self.project_to_function_space(u_alpha_E)
            alpha_N_init = float(self.problem.alpha_N_init)
            alpha_G_init = float(self.problem.alpha_G_init)
            alpha_E_init = 1.0 - alpha_N_init - alpha_G_init
            # calculate charge in volume fractions (alpha) in %
            u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
            u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
            u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
            alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
            alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
            alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

            alpha_Ns.append(alpha_N_diff(point))
            alpha_Gs.append(alpha_G_diff(point))
            alpha_Es.append(alpha_E_diff(point))
            Na_Es.append(Na_E(point))
            K_Es.append(K_E(point))
            Cl_Es.append(Cl_E(point))
            Glu_Ns.append(Glu_N(point))
            Glu_Gs.append(Glu_G(point))
            Glu_Es.append(Glu_E(point))
            phi_Ns.append(phi_N(point))
            phi_Gs.append(phi_G(point))
            phi_Es.append(phi_E(point))

            # get data PDEs
            alpha_N_2 = self.read_from_file_compare(n, 0)
            alpha_G_2 = self.read_from_file_compare(n, 1)
            Na_E_2 = self.read_from_file_compare(n, 4)
            K_E_2 = self.read_from_file_compare(n, 7)
            Cl_E_2 = self.read_from_file_compare(n, 10)
            Glu_N_2 = self.read_from_file_compare(n, 11)
            Glu_G_2 = self.read_from_file_compare(n, 12)
            Glu_E_2 = self.read_from_file_compare(n, 13)
            # concert to mV
            phi_N_2 = self.read_from_file_compare(n, 14, scale=1.0e3)
            phi_G_2 = self.read_from_file_compare(n, 15, scale=1.0e3)
            phi_E_2 = self.read_from_file_compare(n, 16, scale=1.0e3)

            # calculate extracellular volume fraction 2
            u_alpha_E_2 = 1.0 - alpha_N_2 - alpha_G_2
            alpha_E_2 = self.project_to_function_space(u_alpha_E_2)
            alpha_N_init_2 = float(problem_2.alpha_N_init)
            alpha_G_init_2 = float(problem_2.alpha_G_init)
            alpha_E_init_2 = 1.0 - alpha_N_init_2 - alpha_G_init_2
            # calculate charge in volume fractions (alpha) in %
            u_alpha_N_diff_2 = (alpha_N_2 - alpha_N_init_2)/alpha_N_init_2*100
            u_alpha_G_diff_2 = (alpha_G_2 - alpha_G_init_2)/alpha_G_init_2*100
            u_alpha_E_diff_2 = (alpha_E_2 - alpha_E_init_2)/alpha_E_init_2*100
            alpha_N_diff_2 = self.project_to_function_space(u_alpha_N_diff_2)
            alpha_G_diff_2 = self.project_to_function_space(u_alpha_G_diff_2)
            alpha_E_diff_2 = self.project_to_function_space(u_alpha_E_diff_2)

            alpha_Ns_2.append(alpha_N_diff_2(point))
            alpha_Gs_2.append(alpha_G_diff_2(point))
            alpha_Es_2.append(alpha_E_diff_2(point))
            Na_Es_2.append(Na_E_2(point))
            K_Es_2.append(K_E_2(point))
            Cl_Es_2.append(Cl_E_2(point))
            Glu_Ns_2.append(Glu_N_2(point))
            Glu_Gs_2.append(Glu_G_2(point))
            Glu_Es_2.append(Glu_E_2(point))
            phi_Ns_2.append(phi_N_2(point))
            phi_Gs_2.append(phi_G_2(point))
            phi_Es_2.append(phi_E_2(point))

        # range of x values
        xlim = [0.0, Tstop]

        # create plot
        fig = plt.figure(figsize=(19.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,4,1, xlim=xlim, ylim=[0.0, 1.5])
        plt.ylabel(r'[Glu$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        #plt.xticks([0, 25, 50, 75])
        plt.plot(Glu_Es, color='#1f77b4', label=r'Glu', linewidth=lw)
        plt.plot(Glu_Es_2, color='#1f77b4', linestyle='dashed', linewidth=lw)

        ax2 = fig.add_subplot(1,4,2, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'[k$]_e$ (mM)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([20, 40, 60, 80, 100, 120, 140])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(Na_Es, color=c0, label=r'Na$^+$', linewidth=lw)
        plt.plot(K_Es, color=c1,  label=r'K$^+$',linewidth=lw)
        plt.plot(Cl_Es, color=c2, label=r'Cl$^-$',linewidth=lw)
        plt.plot(Na_Es_2, color=c0, linestyle='dashed', linewidth=lw)
        plt.plot(K_Es_2, color=c1, linestyle='dashed', linewidth=lw)
        plt.plot(Cl_Es_2, color=c2, linestyle='dashed', linewidth=lw)

        ax3 = fig.add_subplot(1,4,3, xlim=xlim, ylim=[-100, 20])
        plt.ylabel(r'$\phi$ (mV)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([-90, -70, -50, -30, -10, 10])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(phi_Es, color=c3, label=r'ECS', linewidth=lw)
        plt.plot(phi_Ns, color=c4, label=r'neuron',linewidth=lw)
        plt.plot(phi_Gs, color=c5, label=r'glial', linewidth=lw)
        plt.plot(phi_Es_2, color=c3, linestyle='dashed', linewidth=lw)
        plt.plot(phi_Ns_2, color=c4, linestyle='dashed', linewidth=lw)
        plt.plot(phi_Gs_2, color=c5, linestyle='dashed', linewidth=lw)

        ax4 = fig.add_subplot(1,4,4, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'$\Delta \alpha $ (\%)', fontsize=fosi)
        plt.xlabel(r'time (s)', fontsize=fosi)
        plt.yticks([-40, -30, -20, -10, 0, 10])
        #plt.xticks([0, 25, 50, 75])
        plt.plot(alpha_Es, color=c3, linewidth=lw)
        plt.plot(alpha_Ns, color=c4, linewidth=lw)
        plt.plot(alpha_Gs, color=c5, linewidth=lw)
        plt.plot(alpha_Es_2, color=c3, linestyle='dashed', linewidth=lw)
        plt.plot(alpha_Ns_2, color=c4, linestyle='dashed', linewidth=lw)
        plt.plot(alpha_Gs_2, color=c5, linestyle='dashed', linewidth=lw)

        # make legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.9))

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.35, left=0.1)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{E}', r'\textbf{F}', r'\textbf{G}', r'\textbf{H}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'compare_with_default_time_200'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return

    def make_cl_timeplot(self, path_figs, Tstop):
        """ plot chloride variables at t=Tstop """

        # point at which to calculate duration
        point = 2.0

        # list of function values at point
        alpha_Ns = []; alpha_Gs = []; alpha_Es = []
        Na_Ns = []; K_Ns = []; Cl_Ns = []; Glu_Ns = []
        Na_Gs = []; K_Gs = []; Cl_Gs = []; Glu_Gs = []
        Na_Es = []; K_Es = []; Cl_Es = []; Glu_Es = []
        phi_Ns = []; phi_Gs = []; phi_Es = []

        for n in range(Tstop):

            # get and append data PDEs
            alpha_N = self.read_from_file(n, 0)
            alpha_G = self.read_from_file(n, 1)
            Na_N = self.read_from_file(n, 2)
            Na_G = self.read_from_file(n, 3)
            Na_E = self.read_from_file(n, 4)
            K_N = self.read_from_file(n, 5)
            K_G = self.read_from_file(n, 6)
            K_E = self.read_from_file(n, 7)
            Cl_N = self.read_from_file(n, 8)
            Cl_G = self.read_from_file(n, 9)
            Cl_E = self.read_from_file(n, 10)
            Glu_N = self.read_from_file(n, 11)
            Glu_G = self.read_from_file(n, 12)
            Glu_E = self.read_from_file(n, 13)
            # concert to mV
            phi_N = self.read_from_file(n, 14, scale=1.0e3)
            phi_G = self.read_from_file(n, 15, scale=1.0e3)
            phi_E = self.read_from_file(n, 16, scale=1.0e3)

            # calculate extracellular volume fraction
            u_alpha_E = 1.0 - alpha_N - alpha_G
            alpha_E = self.project_to_function_space(u_alpha_E)
            alpha_N_init = float(self.problem.alpha_N_init)
            alpha_G_init = float(self.problem.alpha_G_init)
            alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

            # calculate charge in volume fractions (alpha) in %
            u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
            u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
            u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
            alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
            alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
            alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

            alpha_Ns.append(alpha_N_diff(point))
            alpha_Gs.append(alpha_G_diff(point))
            alpha_Es.append(alpha_E_diff(point))
            Na_Ns.append(Na_N(point))
            K_Ns.append(K_N(point))
            Cl_Ns.append(Cl_N(point))
            Na_Gs.append(Na_G(point))
            K_Gs.append(K_G(point))
            Cl_Gs.append(Cl_G(point))
            Na_Es.append(Na_E(point))
            K_Es.append(K_E(point))
            Cl_Es.append(Cl_E(point))
            Glu_Ns.append(Glu_N(point))
            Glu_Gs.append(Glu_G(point))
            Glu_Es.append(Glu_E(point))
            phi_Ns.append(phi_N(point))
            phi_Gs.append(phi_G(point))
            phi_Es.append(phi_E(point))

        # range of x values
        xlim = [0.0, Tstop]

        # create plot
        fig = plt.figure(figsize=(19.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,4,1, xlim=xlim, ylim=[0, 150])
        plt.ylabel(r'[Na] (mM)')
        plt.xlabel(r'time (s)')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plt.plot(Na_Es, color=c3, linewidth=lw)
        plt.plot(Na_Ns, color=c4, linewidth=lw)
        plt.plot(Na_Gs, color=c5, linewidth=lw)

        ax2 = fig.add_subplot(1,4,2, xlim=xlim, ylim=[0, 150])
        plt.ylabel(r'[K] (mM)')
        plt.xlabel(r'time (s)')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plt.plot(K_Es, color=c3, linewidth=lw)
        plt.plot(K_Ns, color=c4, linewidth=lw)
        plt.plot(K_Gs, color=c5, linewidth=lw)

        ax3 = fig.add_subplot(1,4,3, xlim=xlim, ylim=[0, 150])
        plt.ylabel(r'[Cl] (mM)')
        plt.xlabel(r'time (s)')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plt.plot(Cl_Es, color=c3, linewidth=lw)
        plt.plot(Cl_Ns, color=c4, linewidth=lw)
        plt.plot(Cl_Gs, color=c5, linewidth=lw)

        ax4 = fig.add_subplot(1,4,4, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'$\Delta \alpha $ (\%)')
        plt.xlabel(r'time (s)')
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plt.plot(alpha_Es, color=c3, label=r'ECS', linewidth=lw)
        plt.plot(alpha_Ns, color=c4, label=r'neuron', linewidth=lw)
        plt.plot(alpha_Gs, color=c5, label=r'glial', linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.35, left=0.1)

        # add legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.9))

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'chloride_timeplot'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()

        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return

    def _tmp_frames(self, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # get data
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        Na_N = self.read_from_file(n, 2)
        Na_G = self.read_from_file(n, 3)
        Na_E = self.read_from_file(n, 4)
        K_N = self.read_from_file(n, 5)
        K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        Cl_N = self.read_from_file(n, 8)
        Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        # convert to uM
        Glu_N = self.read_from_file(n, 11, scale=1.0e3)
        Glu_G = self.read_from_file(n, 12, scale=1.0e3)
        Glu_E = self.read_from_file(n, 13, scale=1.0e3)
        # convert to mV
        phi_N = self.read_from_file(n, 14, scale=1.0e3)
        phi_G = self.read_from_file(n, 15, scale=1.0e3)
        phi_E = self.read_from_file(n, 16, scale=1.0e3)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)
        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # plotting parameters
        xlim = [0.0, 10.0] # range of x values

        # create plot
        fig = plt.figure(figsize=(22.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,6,1, xlim=xlim, ylim=[0, 150])
        plt.title(r'neuron ion concentrations')
        plt.ylabel(r'$[{\rm k}]_n$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_N, color=c0, linewidth=lw)
        plot(K_N, color=c1, linewidth=lw)
        plot(Cl_N, color=c2, linewidth=lw)

        ax1 = fig.add_subplot(1,6,2, xlim=xlim, ylim=[0, 150])
        plt.title(r'glial ion concentrations')
        plt.ylabel(r'$[{\rm k}]_g$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_G, color=c0, linewidth=lw)
        plot(K_G, color=c1, linewidth=lw)
        plot(Cl_G, color=c2, linewidth=lw)
 
        ax2 = fig.add_subplot(1,6,3, xlim=xlim, ylim=[0, 150])
        plt.title(r'ECS ion concentrations')
        plt.ylabel(r'$[{\rm k}]_e$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_E, color=c0, label=r'Na$^+$', linewidth=lw)
        plot(K_E, color=c1, label=r'K$^+$',linewidth=lw)
        plot(Cl_E, color=c2, label=r'Cl$^-$',linewidth=lw)

        ax3 = fig.add_subplot(1,6,4, xlim=xlim, ylim=[-100, 20])
        plt.title(r'potentials')
        plt.ylabel(r'$\phi$ (mV)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_E, color=c3, label=r'ECS', linewidth=lw)
        plot(phi_N, color=c4, label=r'neuron', linewidth=lw)
        plot(phi_G, color=c5, label=r'glial', linewidth=lw)

        ax4 = fig.add_subplot(1,6,5, xlim=xlim, ylim=[-50, 20])
        plt.title(r'change volume fractions')
        plt.ylabel(r'$\Delta \alpha$ (\%)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plot(alpha_E_diff, color=c3, linewidth=lw)
        plot(alpha_N_diff, color=c4, linewidth=lw)
        plot(alpha_G_diff, color=c5, linewidth=lw)

        ax4 = fig.add_subplot(1,6,6, xlim=xlim, ylim=[0, 10])
        plt.title(r'ECS glu')
        plt.ylabel(r'$[{\rm Glu}]_e$ ($\mu$M)')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25])
        plot(Glu_E)

        # add legend
        plt.figlegend(bbox_to_anchor=(0.98, 0.87))

        # make pretty
        ax.axis('off')
        plt.suptitle('t = %.1f s' % float(n), fontsize=20, fontweight='bold', x=0.47)
        plt.subplots_adjust(top=0.85)
        plt.subplots_adjust(wspace=0.4, left=0.035)

        # save figure to file
        fname_res = path_figs + '/_tmp_%05d.png' % n

        plt.savefig(fname_res, format='png')
        plt.close()

    def _tmp_gating(self, path_data, path_figs, n):
        """ plot gating variables at t=n """

        # filename for data ODE
        fname_ODE = path_data + 'ODE/' + 'results.h5'

        # get data
        m_NaT = self.read_from_file(n, 0, fname_compare=fname_ODE)
        h_NaT = self.read_from_file(n, 1, fname_compare=fname_ODE)
        m_NaP = self.read_from_file(n, 2, fname_compare=fname_ODE)
        h_NaP = self.read_from_file(n, 3, fname_compare=fname_ODE)
        m_KDR = self.read_from_file(n, 4, fname_compare=fname_ODE)
        m_KA = self.read_from_file(n, 5, fname_compare=fname_ODE)
        h_KA = self.read_from_file(n, 6, fname_compare=fname_ODE)
        y = self.read_from_file(n, 7, fname_compare=fname_ODE)
        D1 = self.read_from_file(n, 8, fname_compare=fname_ODE)
        D2 = self.read_from_file(n, 9, fname_compare=fname_ODE)

        # range of x values
        xlim = [0.0, 10.0]

        # create plot
        fig = plt.figure(figsize=(22.5*fs, 5*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,5,1, xlim=xlim, ylim=[-0.05, 1.05])
        plt.title(r'NaP')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5.0, 7.5, 10])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plot(m_NaP, linewidth=lw)
        plot(h_NaP, linewidth=lw)

        ax2 = fig.add_subplot(1,5,2, xlim=xlim, ylim=[-0.05, 1.05])
        plt.title(r'KDR')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5.0, 7.5, 10])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plot(m_KDR, linewidth=lw)

        ax3 = fig.add_subplot(1,5,3, xlim=xlim, ylim=[-0.05, 1.05])
        plt.title(r'KA')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5.0, 7.5, 10])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plot(m_KA, linewidth=lw)
        plot(h_KA, linewidth=lw)
        plt.legend([r'm', r'h'], loc='center right')

        ax4 = fig.add_subplot(1,5,4, xlim=xlim, ylim=[-0.05, 1.05])
        plt.title(r'NaT')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5.0, 7.5, 10])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        #plot(m_NaT, linewidth=lw)
        #plot(h_NaT, linewidth=lw)

        ax5 = fig.add_subplot(1,5,5, xlim=xlim, ylim=[-0.05, 1.05])
        plt.title(r'NMDA')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5.0, 7.5, 10])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plot(y, linewidth=lw)
        plot(D1, linewidth=lw)
        plot(D2, linewidth=lw)
        plt.legend([r'y', r'D1', r'D2'], loc='center right')

        # make pretty
        ax.axis('off')
        plt.suptitle('t = %.1f s' % float(n), fontsize=20, fontweight='bold', x=0.47)
        plt.subplots_adjust(top=0.85)
        plt.subplots_adjust(wspace=0.4, left=0.035)

        # save figure to file
        fname_res = path_figs + '_gating_%d' % n
        plt.savefig(fname_res + '.png', format='png')
        plt.close()
        return

    def create_movie(self, res_path):
        """ Save movie and remove frame files """

        print('Making movie animation.mpg - this may take a while')
        filename = res_path + '/movie.mpg'
        try:
            subprocess.call("mencoder 'mf://" + res_path + "/_tmp*.png' -mf type=png:fps=10 -ovc lavc "
                            "-lavcopts vcodec=wmv2 -oac copy -o " + filename, shell=True)
        except:
            print("Making movie failed, not removing tmp image files.")

        return
