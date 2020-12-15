from dolfin import *
import ufl

from problem_base import ProblemBase

class Problem(ProblemBase):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        ProblemBase.__init__(self, mesh, t_PDE, t_ODE)

    def excitatory_currents(self, phi_NE, E_Na_N, E_K_N, E_Cl_N):
        """ Excitatory currents for initiating wave - neuron """
        # set conductance
        Gmax = 5.0          # max conductance (S/m^2)
        LE = 2.0e-5         # stimulate zone at leftmost part of domain
        tE = 2.0            # time of stimuli (s)
        GE = Expression('Gmax*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE, t=self.t_PDE, degree=4)

        # define and return currents
        I_Na_ex = GE*(phi_NE - E_Na_N)  # sodium    - A/m^2
        I_K_ex = GE*(phi_NE - E_K_N)    # potassium - A/m^2
        I_Cl_ex = GE*(phi_NE - E_Cl_N)  # chloride  - A/m^2
        return I_Na_ex, I_K_ex, I_Cl_ex

    def set_membrane_fluxes(self, w, w_, ss_):
        """ set the problems transmembrane ion fluxes. Note that the passive
        fluxes are treated implicitly (w_), while active currents (i.e pumps)
        are treated explicitly (w), except for the gating variables (ss). """
        # get physical parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]

        # split unknowns (PDEs)
        alpha_N, alpha_G, Na_N, Na_G, Na_E, K_N, K_G, K_E, Cl_N, Cl_G, Cl_E, \
                Glu_N, Glu_G, Glu_E, phi_N, phi_G, phi_E = split(w)

        # split solution from previous time step (PDEs)
        alpha_N_, alpha_G_, Na_N_, Na_G_, Na_E_, K_N_, K_G_, K_E_, \
                Cl_N_, Cl_G_, Cl_E_, Glu_N_, Glu_G_, Glu_E_, \
                phi_N_, phi_G_, phi_E_ = split(w_)

        # calculate membrane potentials
        phi_NE = phi_N - phi_E  # neuron (V)
        phi_GE = phi_G - phi_E  # glial  (V)

        ################################################################
        # define Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*ln(Na_E/Na_N) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*ln(K_E/K_N)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_N) # chloride  - (V)
        # define Nernst potential - glial
        E_Na_G = R*temperature/(F*z_Na)*ln(Na_E/Na_G) # sodium    - (V)
        E_K_G = R*temperature/(F*z_K)*ln(K_E/K_G)     # potassium - (V)
        E_Cl_G = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_G) # chloride  - (V)

        ################################################################
        # get currents
        I_NaT, I_NaP, I_KDR, I_KA, I_NMDA = self.voltage_gated_currents(phi_NE, Na_N, Na_E, K_N, K_E, Glu_E, ss_)
        I_ATP_N = self.I_ATP_N(K_E_, Na_N_)
        I_Na_leak_N, I_K_leak_N, I_Cl_leak_N = self.leak_currents_neuron(phi_NE, E_Na_N, E_K_N, E_Cl_N)
        I_Na_leak_G, I_Cl_leak_G = self.leak_currents_glial(phi_GE, E_Na_G, E_Cl_G)
        I_ATP_G = self.I_ATP_G(K_E_, Na_G_)
        I_NaKCl = self.I_NaKCl(Na_G, K_G, Cl_G, Na_E, K_E, Cl_E)
        I_KIR = self.I_KIR(phi_GE, E_K_G, K_G)
        # get excitatory currents
        I_Na_ex, I_K_ex, I_Cl_ex = self.excitatory_currents(phi_NE, E_Na_N, E_K_N, E_Cl_N)

        # Total transmembrane ion currents - neuron
        I_Na_NE = I_Na_leak_N + I_NaP + 3.0*I_ATP_N + 2./3*I_NMDA + I_Na_ex     # sodium    - (A/m^2)
        I_K_NE = I_K_leak_N + I_KDR + I_KA - 2.0*I_ATP_N + 1./3*I_NMDA + I_K_ex # potassium -(A/m^2)
        I_Cl_NE = I_Cl_leak_N + I_Cl_ex                           # chloride  - (A/m^2)

        # total transmembrane ion currents - glial
        I_Na_GE = I_Na_leak_G + 3.0*I_ATP_G + I_NaKCl             # sodium    - (A/m^2)
        I_K_GE = I_KIR - 2.0*I_ATP_G + I_NaKCl                    # potassium - (A/m^2)
        I_Cl_GE = I_Cl_leak_G - 2.0*I_NaKCl                       # chloride  - (A/m^2)

        ################################################################
        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na)      # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)         # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl)      # chloride  - (mol/(m^2s))

        # convert currents currents to flux - glial
        J_Na_GE = I_Na_GE/(F*z_Na)      # sodium    - (mol/(m^2s))
        J_K_GE = I_K_GE/(F*z_K)         # potassium - (mol/(m^2s))
        J_Cl_GE = I_Cl_GE/(F*z_Cl)      # chloride  - (mol/(m^2s))

        ################################################################
        # total glutamate fluxes
        J_Glu_NE, J_Glu_GE = self.J_Glu(phi_NE, Glu_N, Glu_N, Glu_G, Glu_E)

        J_M = [[J_Na_NE, J_Na_GE], \
               [J_K_NE, J_K_GE], \
               [J_Cl_NE, J_Cl_GE], \
               [J_Glu_NE, J_Glu_GE]]

        # set problem's membrane fluxes
        self.membrane_fluxes = J_M
        return

class ProblemStimKCl(ProblemBase):
    """ Problem where CSD wave is initiated by increased ECS K and Cl """
    def __init(self, mesh, t_PDE, t_ODE):
        ProblemBase.__init__(self, mesh, t_PDE, t_ODE)

    def set_initial_conds_PDE(self):
        """ set the PDE problems initial conditions """

        self.alpha_N_init = '0.5'   # volume fraction neuron
        self.alpha_G_init = '0.3'   # volume fraction glial

        self.Na_N_init = '9.3'      # neuron sodium concentration (mol/m^3)
        self.K_N_init = '130'       # neuron potassium concentration (mol/m^3)
        self.Cl_N_init = '8.7'      # neuron chloride concentration (mol/m^3)

        self.Na_G_init = '14'       # glial sodium concentration (mol/m^3)
        self.K_G_init = '130'       # glial potassium concentration (mol/m^3)
        self.Cl_G_init = '8.5'      # glial chloride concentration (mol/m^3)

        self.Na_E_init = '140.5'    # ECS sodium concentration (mol/m^3)
        self.K_E_init = '4'         # ECS potassium concentration (mol/m^3)
        self.Cl_E_init = '113'      # ECS chloride concentration (mol/m^3)

        self.Glu_N_init = '10'      # neuron glutamate concentration (mol/m^3)
        self.Glu_G_init = '10.0e-3' # glial glutamate concentration (mol/m^3)
        self.Glu_E_init = '1.0e-5'  # ECS glutamate concentration (mol/m^3)

        self.phi_N_init = '-0.0685' # neuron potential (V)
        self.phi_G_init = '-0.082'  # neuron potential (V)
        self.phi_E_init = '0.0'     # ECS potential (V)

        self.inits_PDE = Expression((self.alpha_N_init, \
                                     self.alpha_G_init, \
                                     self.Na_N_init, \
                                     self.Na_G_init, \
                                     self.Na_E_init, \
                                     self.K_N_init, \
                                     self.K_G_init, \
                                     self.K_E_init + ' + 12.0*(x[0] < 0.001)', \
                                     self.Cl_N_init, \
                                     self.Cl_G_init, \
                                     self.Cl_E_init + ' + 12.0*(x[0] < 0.001)', \
                                     self.Glu_N_init, \
                                     self.Glu_G_init, \
                                     self.Glu_E_init, \
                                     self.phi_N_init, \
                                     self.phi_G_init, \
                                     self.phi_E_init), degree=4)
        return

class ProblemStimGlu(ProblemBase):
    """ Problem where CSD wave is initiated by increased ECS K and Cl """
    def __init(self, mesh, t_PDE, t_ODE):
        ProblemBase.__init__(self, mesh, t_PDE, t_ODE)

    def set_initial_conds_PDE(self):
        """ set the PDE problems initial conditions """

        self.alpha_N_init = '0.5'   # volume fraction neuron
        self.alpha_G_init = '0.3'   # volume fraction glial

        self.Na_N_init = '9.3'      # neuron sodium concentration (mol/m^3)
        self.K_N_init = '130'       # neuron potassium concentration (mol/m^3)
        self.Cl_N_init = '8.7'      # neuron chloride concentration (mol/m^3)

        self.Na_G_init = '14'       # glial sodium concentration (mol/m^3)
        self.K_G_init = '130'       # glial potassium concentration (mol/m^3)
        self.Cl_G_init = '8.5'      # glial chloride concentration (mol/m^3)

        self.Na_E_init = '140.5'    # ECS sodium concentration (mol/m^3)
        self.K_E_init = '4'         # ECS potassium concentration (mol/m^3)
        self.Cl_E_init = '113'      # ECS chloride concentration (mol/m^3)

        self.Glu_N_init = '10'      # neuron glutamate concentration (mol/m^3)
        self.Glu_G_init = '10.0e-3' # glial glutamate concentration (mol/m^3)
        self.Glu_E_init = '1.0e-5'  # ECS glutamate concentration (mol/m^3)

        self.phi_N_init = '-0.0685' # neuron potential (V)
        self.phi_G_init = '-0.082'  # neuron potential (V)
        self.phi_E_init = '0.0'     # ECS potential (V)

        self.inits_PDE = Expression((self.alpha_N_init, \
                                     self.alpha_G_init, \
                                     self.Na_N_init, \
                                     self.Na_G_init, \
                                     self.Na_E_init, \
                                     self.K_N_init, \
                                     self.K_G_init, \
                                     self.K_E_init, \
                                     self.Cl_N_init, \
                                     self.Cl_G_init, \
                                     self.Cl_E_init , \
                                     self.Glu_N_init, \
                                     self.Glu_G_init, \
                                     self.Glu_E_init + ' + 1.0e-3*(x[0] < 0.001)', \
                                     self.phi_N_init, \
                                     self.phi_G_init, \
                                     self.phi_E_init), degree=4)
        return

class ProblemStimPumpsOff(ProblemBase):
    """ Problem where CSD wave is initiated by turning off pumps """
    def __init(self, mesh, t_PDE, t_ODE):
        ProblemBase.__init__(self, mesh, t_PDE, t_ODE)

    def set_parameters(self):
        """ set the problems physical parameters """
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.332)        # Faraday's constant - (C/mol)
        R = Constant(8.3144598)        # gas constant - (J/(mol*K))

        # membrane parameters
        gamma_NE = Constant(5.3849e5)  # area of membrane per volume - neuron (1/m)
        gamma_GE = Constant(6.3849e5)  # area of membrane per volume - glial  (1/m)

        gamma_M = [gamma_NE, gamma_GE]

        nw_NE = Constant(5.4e-10)      # hydraulic permeability - neuron (m/s/(mol/m^3))
        nw_GE = Constant(5.4e-10)      # hydraulic permeability - glial  (m/s/(mol/m^3))
        nw_M = [nw_NE, nw_GE]

        C_NE = Constant(0.75e-2)       # capacitance - neuron (F/m^2)
        C_GE = Constant(0.75e-2)       # capacitance - glial  (F/m^2)
        C_M = [C_NE, C_GE]

        # ion specific parameters
        D_Na = Constant(1.33e-9)       # diffusion coefficient - sodium (m^2/s)
        D_K = Constant(1.96e-9)        # diffusion coefficient - potassium (m^2/s)
        D_Cl = Constant(2.03e-9)       # diffusion coefficient - chloride (m^2/s)
        D_Glu = Constant(7.6e-10)      # diffusion coefficient - glutamate (m^2/s)
        D = [D_Na, D_K, D_Cl, D_Glu]

        z_Na = Constant(1.0)           # valence - sodium (Na)
        z_K = Constant(1.0)            # valence - potassium (K)
        z_Cl = Constant(-1.0)          # valence - chloride (Cl)
        z_Glu = Constant(0.0)          # valence - chloride (Cl)
        z_0 = Constant(-1.0)           # valence immobile ions
        z = [z_Na, z_K, z_Cl, z_Glu, z_0]

        xie_N = Constant(0.0)          # scaling factor effective diffusion neuron
        xie_G = Constant(0.05)         # scaling factor effective diffusion glial
        xie = [xie_N, xie_G]

        ################################################################
        # permeability for voltage gated membrane currents
        g_NaT = 0.0            # transient Na        - neuron (S/m^2)
        g_NaP = 2.0e-7         # persistent Na       - neuron (S/m^2)
        g_KDR = 1.0e-5         # K delayed rectifier - neuron (S/m^2)
        g_KA  = 1.0e-6         # transient K         - neuron (S/m^2)

        # conductivity for leak currents
        g_Na_leak_N = 2.0e-1   # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = 7.0e-1   # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = 2.0      # chloride (Cl)       - neuron (S/m^2)
        g_Na_leak_G = 7.2e-2   # sodium (Na)         - glial  (S/m^2)
        g_Cl_leak_G = 5.0e-1   # chloride (Cl)       - neuron (S/m^2)

        # other membrane mechanisms
        g_KIR_0 = Constant(1.3)     # K inward rectifier  - glial  (S/m^2)
        g_NaKCl = Constant(8.13e-4) # NaKCl cotransporter - glial  (A/m^2)

        # max pump rate  - glial  (A/m^2)
        I_G = Expression('0.0372*(1 - (x[0]<=0.001)*(t<=2))', t=self.t_PDE, degree=4)
        # max pump rate  - neuron (A/m^2)
        I_N = Expression('0.1372*(1 - (x[0]<=0.001)*(t<=2))', t=self.t_PDE, degree=4)

        m_Na = 7.7  # pump threshold - both   (mol/m^3)
        m_K = 2.0   # pump threshold - both   (mol/m^3)

        # NMDA receptor
        g_NMDA = 1.0e-7        # NMDA permeability   - neuron (S/m^2)
        Mg_E = 2.0             # ECS magnesium       - neuron (mol/m^3)
        k1 = 3.94              # y -> D1             - neuron (1/s)
        k2 = 1.94              # D1 -> y             - neuron (1/s)
        k3 = 0.0213            # D1 -> D2            - neuron (1/s)
        k4 = 0.00277           # D2 -> D1            - neuron (1/s)

        # glutamate cycle parameters
        nu = 0.1               # reabsorbation rate percent
        Ar = 0.1               # release rate - impacts ECS Glu (mol/(m^3s))
        Be = 1.0/42            # decay rate (1/s)
        Bg = 1.0/84            # cycle rate (1/s)
        Rg = 1.0e-3            # glial fraction
        Re = 1.0e-3            # ECS fraction
        eps = 22.99e-3         # saturation constant (mol/m^3)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M, 'xie':xie,
                  'D':D, 'z':z,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N,
                  'g_Na_leak_G':g_Na_leak_G, 'g_Cl_leak_G':g_Cl_leak_G,
                  'g_KDR':g_KDR, 'g_KA':g_KA, 'g_NaP':g_NaP, 'g_NaT':g_NaT,
                  'I_N':I_N, 'I_G':I_G, 'm_K':m_K, 'm_Na':m_Na,
                  'g_KIR_0':g_KIR_0, 'g_NaKCl':g_NaKCl,
                  'nu':nu, 'Ar':Ar, 'Be':Be, 'Bg':Bg, 'Rg':Rg, 'Re':Re,
                  'eps':eps, 'g_NMDA':g_NMDA, 'Mg_E':Mg_E,
                  'k1':k1, 'k2':k2, 'k3':k3, 'k4':k4}

        # set physical parameters
        self.params = params
        # set amount of immobile ions
        self.set_immobile_ions()
        return

class ProblemBlockKIR(Problem):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        Problem.__init__(self, mesh, t_PDE, t_ODE)

    def set_parameters(self):
        """ set the problems physical parameters """
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.332)        # Faraday's constant - (C/mol)
        R = Constant(8.3144598)        # gas constant - (J/(mol*K))

        # membrane parameters
        gamma_NE = Constant(5.3849e5)   # area of membrane per volume - neuron (1/m)
        gamma_GE = Constant(6.3849e5)   # area of membrane per volume - glial  (1/m)

        gamma_M = [gamma_NE, gamma_GE]

        nw_NE = Constant(5.4e-10)       # hydraulic permeability - neuron (m/s/(mol/m^3))
        nw_GE = Constant(5.4e-10)       # hydraulic permeability - glial  (m/s/(mol/m^3))
        nw_M = [nw_NE, nw_GE]

        C_NE = Constant(0.75e-2)        # capacitance - neuron (F/m^2)
        C_GE = Constant(0.75e-2)        # capacitance - glial  (F/m^2)
        C_M = [C_NE, C_GE]

        # ion specific parameters
        D_Na = Constant(1.33e-9)        # diffusion coefficient - sodium (m^2/s)
        D_K = Constant(1.96e-9)         # diffusion coefficient - potassium (m^2/s)
        D_Cl = Constant(2.03e-9)        # diffusion coefficient - chloride (m^2/s)
        D_Glu = Constant(7.6e-10)       # diffusion coefficient - glutamate (m^2/s)
        D = [D_Na, D_K, D_Cl, D_Glu]

        z_Na = Constant(1.0)            # valence - sodium (Na)
        z_K = Constant(1.0)             # valence - potassium (K)
        z_Cl = Constant(-1.0)           # valence - chloride (Cl)
        z_Glu = Constant(0.0)           # valence - chloride (Cl)
        z_0 = Constant(-1.0)            # valence immobile ions
        z = [z_Na, z_K, z_Cl, z_Glu, z_0]

        xie_N = Constant(0.0)           # scaling factor effective diffusion neuron
        xie_G = Constant(0.05)          # scaling factor effective diffusion glial
        xie = [xie_N, xie_G]

        ################################################################
        # permeability for voltage gated membrane currents
        g_NaT = 0.0            # transient Na        - neuron (S/m^2)
        g_NaP = 2.0e-7         # persistent Na       - neuron (S/m^2)
        g_KDR = 1.0e-5         # K delayed rectifier - neuron (S/m^2)
        g_KA  = 1.0e-6         # transient K         - neuron (S/m^2)

        # conductivity for leak currents
        g_Na_leak_N = 2.0e-1   # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = 7.0e-1   # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = 2.0      # chloride (Cl)       - neuron (S/m^2)
        # Updated with block KIR 30% values
        g_Na_leak_G = 2.1e-2   # sodium (Na)         - glial  (S/m^2)
        g_Cl_leak_G = 4.0e-1   # chloride (Cl)       - neuron (S/m^2)

        # other membrane mechanisms
        # Updated with block KIR 30% values
        g_KIR_0_base = 1.3
        g_KIR_0 = Constant(g_KIR_0_base*0.3)     # K inward rectifier  - glial  (S/m^2)
        g_NaKCl = Constant(4.065e-4)             # NaKCl cotransporter - glial  (A/m^2)

        # pump
        # Updated with block KIR 30% values
        I_G = Constant(0.0130) # max pump rate       - glial  (A/m^2)
        I_N = Constant(0.1372) # max pump rate       - neuron (A/m^2)
        m_Na = 7.7             # pump threshold      - both   (mol/m^3)
        m_K = 2.0              # pump threshold      - both   (mol/m^3)

        # NMDA receptor
        g_NMDA = 1.0e-7        # NMDA permeability   - neuron (S/m^2)
        Mg_E = 2.0             # ECS magnesium       - neuron (mol/m^3)
        k1 = 3.94              # y -> D1             - neuron (1/s)
        k2 = 1.94              # D1 -> y             - neuron (1/s)
        k3 = 0.0213            # D1 -> D2            - neuron (1/s)
        k4 = 0.00277           # D2 -> D1            - neuron (1/s)

        # glutamate cycle parameters
        nu = 0.1               # reabsorbation rate percent
        Ar = 0.1               # release rate - impacts ECS Glu (mol/(m^3s))
        Be = 1.0/42            # decay rate (1/s)
        Bg = 1.0/84            # cycle rate (1/s)
        Rg = 1.0e-3            # glial fraction
        Re = 1.0e-3            # ECS fraction
        eps = 22.99e-3         # saturation constant (mol/m^3)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M, 'xie':xie,
                  'D':D, 'z':z,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N,
                  'g_Na_leak_G':g_Na_leak_G, 'g_Cl_leak_G':g_Cl_leak_G,
                  'g_KDR':g_KDR, 'g_KA':g_KA, 'g_NaP':g_NaP, 'g_NaT':g_NaT,
                  'I_N':I_N, 'I_G':I_G, 'm_K':m_K, 'm_Na':m_Na,
                  'g_KIR_0':g_KIR_0, 'g_NaKCl':g_NaKCl,
                  'nu':nu, 'Ar':Ar, 'Be':Be, 'Bg':Bg, 'Rg':Rg, 'Re':Re,
                  'eps':eps, 'g_NMDA':g_NMDA, 'Mg_E':Mg_E,
                  'k1':k1, 'k2':k2, 'k3':k3, 'k4':k4}

        # set physical parameters
        self.params = params
        # calculate and set immobile ions
        self.set_immobile_ions()
        return

    def set_initial_conds_PDE(self):
        """ set the PDE problems initial conditions """

        self.alpha_N_init = '0.5'   # volume fraction neuron
        self.alpha_G_init = '0.3'   # volume fraction glial

        self.Na_N_init = '9.3'      # neuron sodium concentration (mol/m^3)
        self.K_N_init = '130'       # neuron potassium concentration (mol/m^3)
        self.Cl_N_init = '8.7'      # neuron chloride concentration (mol/m^3)

        self.Na_G_init = '14'       # glial sodium concentration (mol/m^3)
        self.K_G_init = '130'       # glial potassium concentration (mol/m^3)
        self.Cl_G_init = '8.5'      # glial chloride concentration (mol/m^3)

        self.Na_E_init = '140.5'    # ECS sodium concentration (mol/m^3)
        self.K_E_init = '4'         # ECS potassium concentration (mol/m^3)
        self.Cl_E_init = '113'      # ECS chloride concentration (mol/m^3)

        self.Glu_N_init = '10'      # neuron glutamate concentration (mol/m^3)
        self.Glu_G_init = '10.0e-3' # glial glutamate concentration (mol/m^3)
        self.Glu_E_init = '1.0e-5'  # ECS glutamate concentration (mol/m^3)

        # steady state with 30 % reduced KIR
        self.phi_N_init = '-0.0685' # neuron potential (V)
        self.phi_G_init = '-0.078'  # neuron potential (V)
        self.phi_E_init = '0.0'     # ECS potential (V)

        self.inits_PDE = Expression((self.alpha_N_init, \
                                     self.alpha_G_init, \
                                     self.Na_N_init, \
                                     self.Na_G_init, \
                                     self.Na_E_init, \
                                     self.K_N_init, \
                                     self.K_G_init, \
                                     self.K_E_init, \
                                     self.Cl_N_init, \
                                     self.Cl_G_init, \
                                     self.Cl_E_init, \
                                     self.Glu_N_init, \
                                     self.Glu_G_init, \
                                     self.Glu_E_init, \
                                     self.phi_N_init, \
                                     self.phi_G_init, \
                                     self.phi_E_init), degree=4)
        return

class ProblemRecurringWaves(ProblemBase):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        ProblemBase.__init__(self, mesh, t_PDE, t_ODE)

    def excitatory_currents(self, phi_NE, E_Na_N, E_K_N, E_Cl_N):
        """ Excitatory currents for initiating wave - neuron """
        # set conductance
        #llim = 85
        llim = 200
        Gmax = 5.0          # max conductance (S/m^2)
        LE = 2.0e-5         # stimulate zone at leftmost part of domain
        tE_1 = 10.0         # time of stimuli #1 (s)
        tE_2 = tE_1 + llim   # time of stimuli #3 (s)
        tE_3 = tE_2 + llim   # time of stimuli #3 (s)
        tE_4 = tE_3 + llim   # time of stimuli #3 (s)
        tE_5 = tE_4 + llim   # time of stimuli #3 (s)

        GE_1 = Expression('Gmax*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t >= tE - 2.0)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE_1, t=self.t_PDE, degree=4)

        GE_2 = Expression('Gmax*100*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t >= tE - 2.0)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE_2, t=self.t_PDE, degree=4)

        GE_3 = Expression('Gmax*100*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t >= tE - 2.0)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE_3, t=self.t_PDE, degree=4)

        GE_4 = Expression('Gmax*100*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t >= tE - 2.0)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE_4, t=self.t_PDE, degree=4)

        GE_5 = Expression('Gmax*100*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t >= tE - 2.0)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE_5, t=self.t_PDE, degree=4)

        GE = GE_1 + GE_2 + GE_3 + GE_4 + GE_5

        # define and return currents
        I_Na_ex = GE*(phi_NE - E_Na_N)  # sodium    - A/m^2
        I_K_ex = GE*(phi_NE - E_K_N)    # potassium - A/m^2
        I_Cl_ex = GE*(phi_NE - E_Cl_N)  # chloride  - A/m^2
        return I_Na_ex, I_K_ex, I_Cl_ex

class ProblemAQP4deletion(Problem):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        Problem.__init__(self, mesh, t_PDE, t_ODE)

    def set_parameters(self):
        """ set the problems physical parameters """
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.332)        # Faraday's constant - (C/mol)
        R = Constant(8.3144598)        # gas constant - (J/(mol*K))

        # membrane parameters
        gamma_NE = Constant(5.3849e5)  # area of membrane per volume - neuron (1/m)
        gamma_GE = Constant(6.3849e5)  # area of membrane per volume - glial  (1/m)

        gamma_M = [gamma_NE, gamma_GE]

        nw_NE = Constant(5.4e-10)      # hydraulic permeability - neuron (m/s/(mol/m^3))
        # reduced by 99.999%
        #nw_GE = Constant(5.4e-15)      # hydraulic permeability - glial  (m/s/(mol/m^3))
        # reduced by 90.0%
        nw_GE = Constant(5.4e-11)      # hydraulic permeability - glial  (m/s/(mol/m^3))
        nw_M = [nw_NE, nw_GE]

        C_NE = Constant(0.75e-2)       # capacitance - neuron (F/m^2)
        C_GE = Constant(0.75e-2)       # capacitance - glial  (F/m^2)
        C_M = [C_NE, C_GE]

        # ion specific parameters
        D_Na = Constant(1.33e-9)       # diffusion coefficient - sodium (m^2/s)
        D_K = Constant(1.96e-9)        # diffusion coefficient - potassium (m^2/s)
        D_Cl = Constant(2.03e-9)       # diffusion coefficient - chloride (m^2/s)
        D_Glu = Constant(7.6e-10)      # diffusion coefficient - glutamate (m^2/s)
        D = [D_Na, D_K, D_Cl, D_Glu]

        z_Na = Constant(1.0)           # valence - sodium (Na)
        z_K = Constant(1.0)            # valence - potassium (K)
        z_Cl = Constant(-1.0)          # valence - chloride (Cl)
        z_Glu = Constant(0.0)          # valence - chloride (Cl)
        z_0 = Constant(-1.0)           # valence immobile ions
        z = [z_Na, z_K, z_Cl, z_Glu, z_0]

        xie_N = Constant(0.0)          # scaling factor effective diffusion neuron
        xie_G = Constant(0.05)         # scaling factor effective diffusion glial
        xie = [xie_N, xie_G]

        ################################################################
        # permeability for voltage gated membrane currents
        g_NaT = 0.0            # transient Na        - neuron (S/m^2)
        g_NaP = 2.0e-7         # persistent Na       - neuron (S/m^2)
        g_KDR = 1.0e-5         # K delayed rectifier - neuron (S/m^2)
        g_KA  = 1.0e-6         # transient K         - neuron (S/m^2)

        # conductivity for leak currents
        g_Na_leak_N = 2.0e-1   # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = 7.0e-1   # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = 2.0      # chloride (Cl)       - neuron (S/m^2)
        g_Na_leak_G = 7.2e-2   # sodium (Na)         - glial  (S/m^2)
        g_Cl_leak_G = 5.0e-1   # chloride (Cl)       - neuron (S/m^2)

        # other membrane mechanisms
        g_KIR_0 = Constant(1.3)     # K inward rectifier  - glial  (S/m^2)
        g_NaKCl = Constant(8.13e-4) # NaKCl cotransporter - glial  (A/m^2)

        # pump
        I_G = Constant(0.0372) # max pump rate       - glial  (A/m^2)
        I_N = Constant(0.1372) # max pump rate       - neuron (A/m^2)
        m_Na = 7.7             # pump threshold      - both   (mol/m^3)
        m_K = 2.0              # pump threshold      - both   (mol/m^3)

        # NMDA receptor
        g_NMDA = 1.0e-7        # NMDA permeability   - neuron (S/m^2)
        Mg_E = 2.0             # ECS magnesium       - neuron (mol/m^3)
        k1 = 3.94              # y -> D1             - neuron (1/s)
        k2 = 1.94              # D1 -> y             - neuron (1/s)
        k3 = 0.0213            # D1 -> D2            - neuron (1/s)
        k4 = 0.00277           # D2 -> D1            - neuron (1/s)

        # glutamate cycle parameters
        nu = 0.1               # reabsorbation rate percent
        Ar = 0.1               # release rate - impacts ECS Glu (mol/(m^3s))
        Be = 1.0/42            # decay rate (1/s)
        Bg = 1.0/84            # cycle rate (1/s)
        Rg = 1.0e-3            # glial fraction
        Re = 1.0e-3            # ECS fraction
        eps = 22.99e-3         # saturation constant (mol/m^3)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M, 'xie':xie,
                  'D':D, 'z':z,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N,
                  'g_Na_leak_G':g_Na_leak_G, 'g_Cl_leak_G':g_Cl_leak_G,
                  'g_KDR':g_KDR, 'g_KA':g_KA, 'g_NaP':g_NaP, 'g_NaT':g_NaT,
                  'I_N':I_N, 'I_G':I_G, 'm_K':m_K, 'm_Na':m_Na,
                  'g_KIR_0':g_KIR_0, 'g_NaKCl':g_NaKCl,
                  'nu':nu, 'Ar':Ar, 'Be':Be, 'Bg':Bg, 'Rg':Rg, 'Re':Re,
                  'eps':eps, 'g_NMDA':g_NMDA, 'Mg_E':Mg_E,
                  'k1':k1, 'k2':k2, 'k3':k3, 'k4':k4}

        # set physical parameters
        self.params = params
        # calculate and set immobile ions
        self.set_immobile_ions()
        return

class ProblemGapJuncGlial(Problem):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        Problem.__init__(self, mesh, t_PDE, t_ODE)

    def set_parameters(self):
        """ set the problems physical parameters """
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.332)        # Faraday's constant - (C/mol)
        R = Constant(8.3144598)        # gas constant - (J/(mol*K))

        # membrane parameters
        gamma_NE = Constant(5.3849e5)  # area of membrane per volume - neuron (1/m)
        gamma_GE = Constant(6.3849e5)  # area of membrane per volume - glial  (1/m)
        gamma_M = [gamma_NE, gamma_GE]

        nw_NE = Constant(5.4e-10)      # hydraulic permeability - neuron (m/s/(mol/m^3))
        nw_GE = Constant(5.4e-10)      # hydraulic permeability - glial  (m/s/(mol/m^3))
        nw_M = [nw_NE, nw_GE]

        C_NE = Constant(0.75e-2)       # capacitance - neuron (F/m^2)
        C_GE = Constant(0.75e-2)       # capacitance - glial  (F/m^2)
        C_M = [C_NE, C_GE]

        # ion specific parameters
        D_Na = Constant(1.33e-9)       # diffusion coefficient - sodium (m^2/s)
        D_K = Constant(1.96e-9)        # diffusion coefficient - potassium (m^2/s)
        D_Cl = Constant(2.03e-9)       # diffusion coefficient - chloride (m^2/s)
        D_Glu = Constant(7.6e-10)      # diffusion coefficient - glutamate (m^2/s)
        D = [D_Na, D_K, D_Cl, D_Glu]

        z_Na = Constant(1.0)           # valence - sodium (Na)
        z_K = Constant(1.0)            # valence - potassium (K)
        z_Cl = Constant(-1.0)          # valence - chloride (Cl)
        z_Glu = Constant(0.0)          # valence - chloride (Cl)
        z_0 = Constant(-1.0)           # valence immobile ions
        z = [z_Na, z_K, z_Cl, z_Glu, z_0]

        xie_N = Constant(0.0)          # scaling factor effective diffusion neuron
        xie_G = Constant(0.0)          # scaling factor effective diffusion glial
        xie = [xie_N, xie_G]

        ################################################################
        # permeability for voltage gated membrane currents
        g_NaT = 0.0            # transient Na        - neuron (S/m^2)
        g_NaP = 2.0e-7         # persistent Na       - neuron (S/m^2)
        g_KDR = 1.0e-5         # K delayed rectifier - neuron (S/m^2)
        g_KA  = 1.0e-6         # transient K         - neuron (S/m^2)

        # conductivity for leak currents
        g_Na_leak_N = 2.0e-1   # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = 7.0e-1   # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = 2.0      # chloride (Cl)       - neuron (S/m^2)
        g_Na_leak_G = 7.2e-2   # sodium (Na)         - glial  (S/m^2)
        g_Cl_leak_G = 5.0e-1   # chloride (Cl)       - neuron (S/m^2)

        # other membrane mechanisms
        g_KIR_0 = Constant(1.3)     # K inward rectifier  - glial  (S/m^2)
        g_NaKCl = Constant(8.13e-4) # NaKCl cotransporter - glial  (A/m^2)

        # pump
        I_G = Constant(0.0372) # max pump rate       - glial  (A/m^2)
        I_N = Constant(0.1372) # max pump rate       - neuron (A/m^2)
        m_Na = 7.7             # pump threshold      - both   (mol/m^3)
        m_K = 2.0              # pump threshold      - both   (mol/m^3)

        # NMDA receptor
        g_NMDA = 1.0e-7        # NMDA permeability   - neuron (S/m^2)
        Mg_E = 2.0             # ECS magnesium       - neuron (mol/m^3)
        k1 = 3.94              # y -> D1             - neuron (1/s)
        k2 = 1.94              # D1 -> y             - neuron (1/s)
        k3 = 0.0213            # D1 -> D2            - neuron (1/s)
        k4 = 0.00277           # D2 -> D1            - neuron (1/s)

        # glutamate cycle parameters
        nu = 0.1               # reabsorbation rate percent
        Ar = 0.1               # release rate - impacts ECS Glu (mol/(m^3s))
        Be = 1.0/42            # decay rate (1/s)
        Bg = 1.0/84            # cycle rate (1/s)
        Rg = 1.0e-3            # glial fraction
        Re = 1.0e-3            # ECS fraction
        eps = 22.99e-3         # saturation constant (mol/m^3)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M, 'xie':xie,
                  'D':D, 'z':z,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N,
                  'g_Na_leak_G':g_Na_leak_G, 'g_Cl_leak_G':g_Cl_leak_G,
                  'g_KDR':g_KDR, 'g_KA':g_KA, 'g_NaP':g_NaP, 'g_NaT':g_NaT,
                  'I_N':I_N, 'I_G':I_G, 'm_K':m_K, 'm_Na':m_Na,
                  'g_KIR_0':g_KIR_0, 'g_NaKCl':g_NaKCl,
                  'nu':nu, 'Ar':Ar, 'Be':Be, 'Bg':Bg, 'Rg':Rg, 'Re':Re,
                  'eps':eps, 'g_NMDA':g_NMDA, 'Mg_E':Mg_E,
                  'k1':k1, 'k2':k2, 'k3':k3, 'k4':k4}

        # set physical parameters
        self.params = params
        # calculate and set immobile ions
        self.set_immobile_ions()
        return

class ProblemNewGammas(Problem):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        Problem.__init__(self, mesh, t_PDE, t_ODE)

    def set_parameters(self):
        """ set the problems physical parameters """
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.332)        # Faraday's constant - (C/mol)
        R = Constant(8.3144598)        # gas constant - (J/(mol*K))

        # membrane parameters
        gamma_NE = Constant(1.85e5)    # area of membrane per volume - neuron (1/m)
        gamma_GE = Constant(2.0e5)     # area of membrane per volume - glial  (1/m)
        gamma_M = [gamma_NE, gamma_GE]

        nw_NE = Constant(5.4e-10)      # hydraulic permeability - neuron (m/s/(mol/m^3))
        nw_GE = Constant(5.4e-10)      # hydraulic permeability - glial  (m/s/(mol/m^3))
        nw_M = [nw_NE, nw_GE]

        C_NE = Constant(0.75e-2)       # capacitance - neuron (F/m^2)
        C_GE = Constant(0.75e-2)       # capacitance - glial  (F/m^2)
        C_M = [C_NE, C_GE]

        # ion specific parameters
        D_Na = Constant(1.33e-9)       # diffusion coefficient - sodium (m^2/s)
        D_K = Constant(1.96e-9)        # diffusion coefficient - potassium (m^2/s)
        D_Cl = Constant(2.03e-9)       # diffusion coefficient - chloride (m^2/s)
        D_Glu = Constant(7.6e-10)      # diffusion coefficient - glutamate (m^2/s)
        D = [D_Na, D_K, D_Cl, D_Glu]

        z_Na = Constant(1.0)           # valence - sodium (Na)
        z_K = Constant(1.0)            # valence - potassium (K)
        z_Cl = Constant(-1.0)          # valence - chloride (Cl)
        z_Glu = Constant(0.0)          # valence - chloride (Cl)
        z_0 = Constant(-1.0)           # valence immobile ions
        z = [z_Na, z_K, z_Cl, z_Glu, z_0]

        xie_N = Constant(0.0)          # scaling factor effective diffusion neuron
        xie_G = Constant(0.05)         # scaling factor effective diffusion glial
        xie = [xie_N, xie_G]

        ################################################################
        # permeability for voltage gated membrane currents
        g_NaT = 0.0            # transient Na        - neuron (S/m^2)
        g_NaP = 2.0e-7         # persistent Na       - neuron (S/m^2)
        g_KDR = 1.0e-5         # K delayed rectifier - neuron (S/m^2)
        g_KA  = 1.0e-6         # transient K         - neuron (S/m^2)

        # conductivity for leak currents
        g_Na_leak_N = 2.0e-1   # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = 7.0e-1   # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = 2.0      # chloride (Cl)       - neuron (S/m^2)
        g_Na_leak_G = 7.2e-2   # sodium (Na)         - glial  (S/m^2)
        g_Cl_leak_G = 5.0e-1   # chloride (Cl)       - neuron (S/m^2)

        # other membrane mechanisms
        g_KIR_0 = Constant(1.3)     # K inward rectifier  - glial  (S/m^2)
        g_NaKCl = Constant(8.13e-4) # NaKCl cotransporter - glial  (A/m^2)

        # pump
        I_G = Constant(0.0372) # max pump rate       - glial  (A/m^2)
        I_N = Constant(0.1372) # max pump rate       - neuron (A/m^2)
        m_Na = 7.7             # pump threshold      - both   (mol/m^3)
        m_K = 2.0              # pump threshold      - both   (mol/m^3)

        # NMDA receptor
        g_NMDA = 1.0e-7        # NMDA permeability   - neuron (S/m^2)
        Mg_E = 2.0             # ECS magnesium       - neuron (mol/m^3)
        k1 = 3.94              # y -> D1             - neuron (1/s)
        k2 = 1.94              # D1 -> y             - neuron (1/s)
        k3 = 0.0213            # D1 -> D2            - neuron (1/s)
        k4 = 0.00277           # D2 -> D1            - neuron (1/s)

        # glutamate cycle parameters
        nu = 0.1               # reabsorbation rate percent
        Ar = 0.1               # release rate - impacts ECS Glu (mol/(m^3s))
        Be = 1.0/42            # decay rate (1/s)
        Bg = 1.0/84            # cycle rate (1/s)
        Rg = 1.0e-3            # glial fraction
        Re = 1.0e-3            # ECS fraction
        eps = 22.99e-3         # saturation constant (mol/m^3)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M, 'xie':xie,
                  'D':D, 'z':z,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N,
                  'g_Na_leak_G':g_Na_leak_G, 'g_Cl_leak_G':g_Cl_leak_G,
                  'g_KDR':g_KDR, 'g_KA':g_KA, 'g_NaP':g_NaP, 'g_NaT':g_NaT,
                  'I_N':I_N, 'I_G':I_G, 'm_K':m_K, 'm_Na':m_Na,
                  'g_KIR_0':g_KIR_0, 'g_NaKCl':g_NaKCl,
                  'nu':nu, 'Ar':Ar, 'Be':Be, 'Bg':Bg, 'Rg':Rg, 'Re':Re,
                  'eps':eps, 'g_NMDA':g_NMDA, 'Mg_E':Mg_E,
                  'k1':k1, 'k2':k2, 'k3':k3, 'k4':k4}

        # set physical parameters
        self.params = params
        # calculate and set immobile ions
        self.set_immobile_ions()
        return

    def set_initial_conds_PDE(self):
        """ set the PDE problems initial conditions """
        self.alpha_N_init = '0.5'   # volume fraction neuron
        self.alpha_G_init = '0.3'   # volume fraction glial

        self.Na_N_init = '9.3'      # neuron sodium concentration (mol/m^3)
        self.K_N_init = '130'       # neuron potassium concentration (mol/m^3)
        self.Cl_N_init = '8.7'      # neuron chloride concentration (mol/m^3)

        self.Na_G_init = '14'       # glial sodium concentration (mol/m^3)
        self.K_G_init = '130'       # glial potassium concentration (mol/m^3)
        self.Cl_G_init = '8.5'      # glial chloride concentration (mol/m^3)

        self.Na_E_init = '140.5'    # ECS sodium concentration (mol/m^3)
        self.K_E_init = '4'         # ECS potassium concentration (mol/m^3)
        self.Cl_E_init = '113'      # ECS chloride concentration (mol/m^3)

        self.Glu_N_init = '10'      # neuron glutamate concentration (mol/m^3)
        self.Glu_G_init = '10.0e-3' # glial glutamate concentration (mol/m^3)
        self.Glu_E_init = '1.0e-5'  # ECS glutamate concentration (mol/m^3)

        self.phi_N_init = '-0.0685' # neuron potential (V)
        self.phi_G_init = '-0.082'  # neuron potential (V)
        self.phi_E_init = '0.0'     # ECS potential (V)

        self.inits_PDE = Expression((self.alpha_N_init, \
                                     self.alpha_G_init, \
                                     self.Na_N_init, \
                                     self.Na_G_init, \
                                     self.Na_E_init, \
                                     self.K_N_init, \
                                     self.K_G_init, \
                                     self.K_E_init, \
                                     self.Cl_N_init, \
                                     self.Cl_G_init, \
                                     self.Cl_E_init, \
                                     self.Glu_N_init, \
                                     self.Glu_G_init, \
                                     self.Glu_E_init, \
                                     self.phi_N_init, \
                                     self.phi_G_init, \
                                     self.phi_E_init), degree=4)
        return
