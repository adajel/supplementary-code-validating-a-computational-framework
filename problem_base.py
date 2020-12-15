from dolfin import *
import ufl

class ProblemBase():
    """ Problem base with default parameter values converted to SI units
        from [Mori 2015, A multidomain model for ionic electrodiffusion and
        osmosis with an application to spreading depression] with
        3 compartments: neuron, glial and ECS.

        No initiation of CSD wave in problem base class. """

    def __init__(self, mesh, boundary_point, t_PDE, t_ODE):
        # mesh
        self.mesh = mesh
        # time constant (for updating source and boundary terms)
        self.t_PDE = t_PDE
        self.t_ODE = t_ODE
        self.N_ions = 4     # number of ions
        self.N_comparts = 3 # number of compartments
        self.N_states = 10  # number of states

        # point at which to set phi_E to zero
        self.boundary_point = boundary_point

        # set parameters and initial conditions
        self.set_initial_conds_PDE()
        self.set_parameters()
        self.set_immobile_ions()
        self.set_initial_conds_ODE()
        return

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

    def set_initial_conds_ODE(self):
        """ set the ODE problems initial conditions """

        # get initial membrane potential
        phi_NE_init = float(self.phi_N_init) - float(self.phi_E_init)
        # get initial ECS glutamate concentration
        Glu_E_init = float(self.Glu_E_init)

        # set initial conditions for ODEs
        m_NaT_init = 'alpha_m_NaT/(alpha_m_NaT + beta_m_NaT)' # NaT activation
        h_NaT_init = 'alpha_h_NaT/(alpha_h_NaT + beta_h_NaT)' # NaT inactivation
        m_NaP_init = 'alpha_m_NaP/(alpha_m_NaP + beta_m_NaP)' # NaP activation
        h_NaP_init = 'alpha_h_NaP/(alpha_h_NaP + beta_h_NaP)' # NaP inactivation
        m_KDR_init = 'alpha_m_KDR/(alpha_m_KDR + beta_m_KDR)' # KDR activation
        m_KA_init = 'alpha_m_KA/(alpha_m_KA + beta_m_KA)'     # KA activation
        h_KA_init = 'alpha_h_KA/(alpha_h_KA + beta_h_KA)'     # KA inactivation
        y_init = '(k2*k4)/(k1*k3+k1*k4+k2*k4)'                # NMDA
        D1_init = '(k1*k4)/(k1*k3+k1*k4+k2*k4)'               # NMDA
        D2_init = '(k1*k3)/(k1*k3+k1*k4+k2*k4)'               # NMDA

        self.inits_ODE = Expression((m_NaT_init, h_NaT_init, \
                                     m_NaP_init, h_NaP_init, \
                                     m_KDR_init, \
                                     m_KA_init, h_KA_init, \
                                     y_init, D1_init, D2_init),
                                     alpha_m_NaT=self.alpha_m_NaT(phi_NE_init),
                                     beta_m_NaT=self.beta_m_NaT(phi_NE_init),
                                     alpha_h_NaT=self.alpha_h_NaT(phi_NE_init),
                                     beta_h_NaT=self.beta_h_NaT(phi_NE_init),
                                     alpha_m_NaP=self.alpha_m_NaP(phi_NE_init),
                                     beta_m_NaP=self.beta_m_NaP(phi_NE_init),
                                     alpha_h_NaP=self.alpha_h_NaP(phi_NE_init),
                                     beta_h_NaP=self.beta_h_NaP(phi_NE_init),
                                     alpha_m_KDR=self.alpha_m_KDR(phi_NE_init),
                                     beta_m_KDR=self.beta_m_KDR(phi_NE_init),
                                     alpha_m_KA=self.alpha_m_KA(phi_NE_init),
                                     beta_m_KA=self.beta_m_KA(phi_NE_init),
                                     alpha_h_KA=self.alpha_h_KA(phi_NE_init),
                                     beta_h_KA=self.beta_h_KA(phi_NE_init),
                                     k1 = self.params['k1']*self.F_Glu(Glu_E_init),
                                     k2 = self.params['k2'],
                                     k3 = self.params['k3'],
                                     k4 = self.params['k4'],
                                     degree=4)
        return

    def set_immobile_ions(self):
        """ calculate and set amount of immobile ions """
        # get physical parameters
        F = self.params['F']
        gamma_NE = self.params['gamma_M'][0]
        gamma_GE = self.params['gamma_M'][1]
        C_NE = self.params['C_M'][0]
        C_GE = self.params['C_M'][1]
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_0 = self.params['z'][4]

        # get initial membrane potential
        phi_NE_init = float(self.phi_N_init) - float(self.phi_E_init)
        phi_GE_init = float(self.phi_G_init) - float(self.phi_E_init)
        # get initial volume fractions
        alpha_N_init = float(self.alpha_N_init)
        alpha_G_init = float(self.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        # amount of immobile ions neuron (mol/m^3)
        a_N = 1.0/(z_0*F)*gamma_NE*C_NE*phi_NE_init \
            - 1.0/z_0*alpha_N_init*(z_Na*float(self.Na_N_init) \
                                  + z_K*float(self.K_N_init) \
                                  + z_Cl*float(self.Cl_N_init))
        # amount of immobile ions glial (mol/m^3)
        a_G = 1.0/(z_0*F)*gamma_GE*C_GE*phi_GE_init \
            - 1.0/z_0*alpha_G_init*(z_Na*float(self.Na_G_init) \
                                  + z_K*float(self.K_G_init) \
                                  + z_Cl*float(self.Cl_G_init))
        # amount of immobile ions ECS (mol/m^3)
        a_E = - 1.0/(z_0*F)*gamma_NE*C_NE*phi_NE_init \
              - 1.0/(z_0*F)*gamma_GE*C_GE*phi_GE_init \
              - 1.0/z_0*alpha_E_init*(z_Na*float(self.Na_E_init) \
                                    + z_K*float(self.K_E_init) \
                                    + z_Cl*float(self.Cl_E_init))

        # set amount of immobile ions (mol/m^3)
        self.params['a'] = [a_N, a_G, a_E]
        return

    def voltage_gated_currents(self, phi_NE, Na_N, Na_E, K_N, K_E, Glu_E, ss):
        """ Voltage gated currents - neuron (I_NaT, I_NaP, I_KDR, I_KA, I_NMDA) """
        # get physical parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        g_NaT = self.params['g_NaT']
        g_NaP = self.params['g_NaP']
        g_KDR = self.params['g_KDR']
        g_KA = self.params['g_KA']
        g_NMDA = self.params['g_NMDA']
        Mg_E = self.params['Mg_E']

        # split previous solution (ODEs)
        m_NaT_, h_NaT_, m_NaP_, h_NaP_, m_KDR_, m_KA_, h_KA_, y_, D1_, D2_ = split(ss)

        # define Goldman-Hodgkin-Katz driving force
        mu = F*phi_NE/(R*temperature)                     # shorthand
        # define Goldman Hodgkin Katz driving force
        GHK_Na = mu*F*(Na_N*exp(mu) - Na_E)/(exp(mu) - 1) # sodium    - (C/m^3)
        GHK_K = mu*F*(K_N*exp(mu) - K_E)/(exp(mu) - 1)    # potassium - (C/m^3)

        # NMDA receptor conductivity
        F_Glu = self.F_Glu(Glu_E)
        G_Glu = self.G_Glu(phi_NE, Mg_E)
        I_NMDA_Na = g_NMDA*GHK_Na
        I_NMDA_K = g_NMDA*GHK_K

        # define and return currents
        I_NaT = g_NaT*m_NaT_**3*h_NaT_*GHK_Na                    # transient sodium (NaT)            - (A/m^2)
        I_NaP = g_NaP*m_NaP_**2*h_NaP_*GHK_Na                    # persistent sodium (NaP)           - (A/m^2)
        I_KDR = g_KDR*m_KDR_**2*GHK_K                            # potassium delayed rectifier (KDR) - (A/m^2)
        I_KA = g_KA*m_KA_**2*h_KA_*GHK_K                         # transient potassium (KA)          - (A/m^2)
        I_NMDA = G_Glu*F_Glu*y_*(2./3*I_NMDA_Na + 1./3*I_NMDA_K) # NMDA                              - (A/m^2)

        return I_NaT, I_NaP, I_KDR, I_KA, I_NMDA

    def leak_currents_neuron(self, phi_NE, E_Na_N, E_K_N, E_Cl_N):
        """ Leak currents - neuron """
        # get physical parameters
        g_Na_leak_N = self.params['g_Na_leak_N']
        g_K_leak_N = self.params['g_K_leak_N']
        g_Cl_leak_N = self.params['g_Cl_leak_N']

        # define and return currents
        I_Na_leak_N = g_Na_leak_N*(phi_NE - E_Na_N)  # sodium    - (A/m^2)
        I_K_leak_N = g_K_leak_N*(phi_NE - E_K_N)     # potassium - (A/m^2)
        I_Cl_leak_N = g_Cl_leak_N*(phi_NE - E_Cl_N)  # chloride  - (A/m^2)
        return I_Na_leak_N, I_K_leak_N, I_Cl_leak_N

    def leak_currents_glial(self, phi_GE, E_Na_G, E_Cl_G):
        """ Leak currents - glial """
        # get physical parameters
        g_Na_leak_G = self.params['g_Na_leak_G']
        g_Cl_leak_G = self.params['g_Cl_leak_G']

        # define and return currents
        I_Na_leak_G = g_Na_leak_G*(phi_GE - E_Na_G)  # sodium    - (A/m^2)
        I_Cl_leak_G = g_Cl_leak_G*(phi_GE - E_Cl_G)  # chloride  - (A/m^2)
        return I_Na_leak_G, I_Cl_leak_G

    def I_KIR(self, phi_GE, E_K_G, K_G):
        """ Potassium inward rectifier (KIR) current - glial """
        # get physical parameters
        g_KIR_0 = self.params['g_KIR_0']
        K_G_init = float(self.K_G_init)

        # set conductance
        A = 18.5/42.5                               # shorthand
        B = 1.0e3*(phi_GE - E_K_G + 18.5e-3)/42.5   # shorthand
        C = (-118.6 - 85.2)/44.1                    # shorthand
        D = 1.0e3*(-118.6e-3 + phi_GE)/44.1         # shorthand
        g_KIR = sqrt(K_G/K_G_init)*(1 + exp(A))/(1 + exp(B))*\
                                   (1 + exp(C))/(1 + exp(D))

        # define and return current
        I_KIR = g_KIR_0*g_KIR*(phi_GE - E_K_G)      # A/m^2
        return I_KIR

    def I_ATP_N(self, K_E_, Na_N_):
        """ Na/K ATPase pump current - neuron """
        # get physical parameters
        m_Na = self.params['m_Na']
        m_K = self.params['m_K']
        I_N = self.params['I_N']

        # define and return current
        I_ATP_N = I_N/((1.0 + m_K/K_E_)**2*(1 + m_Na/Na_N_)**3) # A/m^2
        return I_ATP_N

    def I_ATP_G(self, K_E_, Na_G_):
        """ Na/K ATPase pump current - glial """
        # get physical parameters
        m_Na = self.params['m_Na']
        m_K = self.params['m_K']
        I_G = self.params['I_G']

        # define and return current
        I_ATP_G = I_G/((1.0 + m_K/K_E_)**2*(1 + m_Na/Na_G_)**3) # A/m^2
        return I_ATP_G

    def I_NaKCl(self, Na_G, K_G, Cl_G, Na_E, K_E, Cl_E):
        """ NaKCl cotransporter current - glial """
        # get physical parameters
        g_NaKCl = self.params['g_NaKCl']

        # define and return current
        I_NaKCl = g_NaKCl*ln((Na_G*K_G*Cl_G**2)/(Na_E*K_E*Cl_E**2)) # A/m^2
        return I_NaKCl

    def J_Glu(self, phi_NE, Glu_N, Glu_N_, Glu_G_, Glu_E_):
        """ total glutamate fluxes over membrane """
        # get physical parameters
        R = self.params['R']
        F = self.params['F']
        temperature = self.params['temperature']
        Ar = self.params['Ar']
        eps = self.params['eps']
        Rg = self.params['Rg']
        Re = self.params['Re']
        Bg = self.params['Bg']
        Be = self.params['Be']
        nu = self.params['nu']

        # synaptic release of glutamate
        V = 1.0e3*(phi_NE - 8.66e-3)                  # convert from V to mV
        f_syn = 0.76*exp(-0.0044*V**2)                # mol/(m^3)
        # synaptic release flux
        # TODO - whats the units here????
        J_Glu_syn = Ar*Glu_N/(Glu_N + eps)*f_syn      # mol/(m^3s)(mol/(m^3))

        # glutamate fluxes between compartments
        J_Glu_NtoE = nu*Be*(Glu_E_ - Re*Rg*Glu_N_)    # N -> E - mol/(m^3s)
        J_Glu_GtoE = (1 - nu)*Be*(Glu_E_ - Re*Glu_G_) # G -> E - mol/(m^3s)
        J_Glu_NtoG = Bg*(Glu_G_ - Rg*Glu_N_)          # N -> G - mol/(m^3s)

        # define and return membrane fluxes
        J_Glu_NE = - J_Glu_syn + J_Glu_NtoE + J_Glu_NtoG # neuron - (mol/(m^3s))
        J_Glu_GE = J_Glu_GtoE - J_Glu_NtoG               # glial  - (mol/(m^3s))
        return J_Glu_NE, J_Glu_GE

    def set_membrane_fluxes(self, w, w_, ss_):
        """ set the problems transmembrane ion fluxes. Note that the passive
        fluxes are treated implicitly (w_), while active currents (i.e. pumps)
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

        # Total transmembrane ion currents - neuron
        I_Na_NE = I_Na_leak_N + I_NaT + I_NaP + 3.0*I_ATP_N + 2./3*I_NMDA # sodium    - (A/m^2)
        I_K_NE = I_K_leak_N + I_KDR + I_KA - 2.0*I_ATP_N + 1./3*I_NMDA    # potassium -(A/m^2)
        I_Cl_NE = I_Cl_leak_N                                             # chloride  - (A/m^2)

        # total transmembrane ion currents - glial
        I_Na_GE = I_Na_leak_G + 3.0*I_ATP_G + I_NaKCl # sodium    - (A/m^2)
        I_K_GE = I_KIR - 2.0*I_ATP_G + I_NaKCl        # potassium - (A/m^2)
        I_Cl_GE = I_Cl_leak_G - 2.0*I_NaKCl           # chloride  - (A/m^2)

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

    def alpha_m_NaT(self, phi_NE):
        """ transient sodium (NaT), forward rate for activation """
        # convert from V to mV
        V = 1000*(phi_NE + 51.9e-3)
        alpha_m_NaT = 0.32*V/(1 - exp(-0.25*V))
        # convert from 1/ms to 1/s
        return alpha_m_NaT*1.0e3

    def beta_m_NaT(self, phi_NE):
        """ transient sodium (NaT) - backward rate activation """
        # convert to mV
        V = 1000*(phi_NE + 24.89e-3)
        beta_m_NaT = 0.28*V/(exp(0.2*V) - 1)
        # convert from 1/ms to 1/s
        return beta_m_NaT*1.0e3

    def alpha_h_NaT(self, phi_NE):
        """ transient sodium (NaT), forward rate for activation """
        # convert from V to mV
        V = 1000*(0.056*phi_NE + 2.94e-3)
        alpha_h_NaT = 0.128*exp(-V)
        # convert from 1/ms to 1/s
        return alpha_h_NaT*1.0e3

    def beta_h_NaT(self, phi_NE):
        """ transient sodium (NaT) - backward rate activation """
        # convert to mV
        V = 1000*(0.2*phi_NE + 6.0e-3)
        beta_h_NaT = 4.0/(1 + exp(-V))
        # convert from 1/ms to 1/s
        return beta_h_NaT*1.0e3

    def alpha_m_NaP(self, phi_NE):
        """ persistent sodium (NaP), forward rate for activation """
        # convert from V to mV
        V = 1.0e3*(0.143*phi_NE + 5.67e-3)
        alpha_m_NaP = 1.0/(1 + exp(-V))/6.0
        # convert from 1/ms to 1/s
        return alpha_m_NaP*1.0e3

    def beta_m_NaP(self, phi_NE):
        """ persistent sodium (NaP) - backward rate activation """
        # convert to mV
        beta_m_NaP = 1.0/6.0 - self.alpha_m_NaP(phi_NE)/1.0e3
        # convert from 1/ms to 1/s
        return beta_m_NaP*1.0e3

    def alpha_h_NaP(self, phi_NE):
        """ persistent sodium (NaP) - forward rate inactivation """
        # convert from V to mV
        V = 1.0e3*(0.056*phi_NE + 2.94e-3)
        alpha_h_NaP = 5.12e-6*exp(-V)
        # convert from 1/ms to 1/s
        return alpha_h_NaP*1.0e3

    def beta_h_NaP(self, phi_NE):
        """ persistent sodium (NaP) - backward rate inactivation """
        # convert from V to mV
        V = 1.0e3*(0.2*phi_NE + 8.0e-3)
        beta_h_NaP = 1.6e-4/(1 + exp(-V))
        # convert from 1/ms to 1/s
        return beta_h_NaP*1.0e3

    def alpha_m_KDR(self, phi_NE):
        """ potassium delayed rectifier (KDR) - forward rate activation """
        # convert from V to mV
        V = 1.0e3*(phi_NE + 34.9e-3)
        alpha_m_KDR = 0.016*V/(1 - exp(-0.2*V))
        # convert from 1/ms to 1/s
        return alpha_m_KDR*1.0e3

    def beta_m_KDR(self, phi_NE):
        """ potassium delayed rectifier (KDR) - backward rate activation """
        # convert from V to mV
        V = 1.0e3*(0.025*phi_NE + 1.25e-3)
        beta_m_KDR = 0.25*exp(-V)
        # convert from 1/ms to 1/s
        return beta_m_KDR*1.0e3

    def alpha_m_KA(self, phi_NE):
        """ transient potassium (KA) - forward rate activation """
        # convert from V to mV
        V = 1.0e3*(phi_NE + 56.9e-3)
        alpha_m_KA = 0.02*V/(1 - exp(-0.1*V))
        # convert from 1/ms to 1/s
        return alpha_m_KA*1.0e3

    def beta_m_KA(self, phi_NE):
        """ transient potassium (KA) - backward rate activation """
        # convert from V to mV
        V = 1.0e3*(phi_NE + 29.9e-3)
        beta_m_KA = 0.0175*V/(exp(0.1*V) - 1)
        # convert from 1/ms to 1/s
        return beta_m_KA*1.0e3

    def alpha_h_KA(self, phi_NE):
        """ transient potassium (KA) - forward rate inactivation """
        # convert from V to mV
        V = 1.0e3*(0.056*phi_NE + 4.61e-3)
        alpha_h_KA = 0.016*exp(-V)
        # convert from 1/ms to 1/s
        return alpha_h_KA*1.0e3

    def beta_h_KA(self, phi_NE):
        """ transient potassium (KA) - backward rate inactivation """
        # convert from V to mV
        V = 1.0e3*(0.2*phi_NE + 11.98e-3)
        beta_h_KA = 0.5/(1 + exp(-V))
        # convert from 1/ms to 1/s
        return beta_h_KA*1.0e3

    def F_Glu(self, Glu_E):
        """ Glu_E dependence: N-methyl-D-aspartate receptor(NMDA) """
        F_Glu = Glu_E**(1.5)/(Glu_E**(1.5) + (2.3e-3)**(1.5))
        return F_Glu

    def G_Glu(self, phi_NE, Mg_E):
        """ Voltage dependence: N-methyl-D-aspartate receptor(NMDA) """
        # convert from V to mV
        V = 1.0e3*(0.062*phi_NE)
        G_Glu = 1.0/(1 + 0.56*exp(-V)*(Mg_E/3.57))
        return G_Glu

    def F(self, w_, ss, time=None):
        """ Right hand side of the ODE system """
        time = time if time else Constant(0.0)

        k1 = self.params['k1']
        k2 = self.params['k2']
        k3 = self.params['k3']
        k4 = self.params['k4']

        Glu_E_ = split(w_)[self.N_comparts*(1 + self.N_ions) - 2]

        # split function for unknown PDE solution in previous time step
        phi_N_ = split(w_)[self.N_comparts*(1 + self.N_ions) - 2 + 1]
        phi_E_ = split(w_)[self.N_comparts*(1 + self.N_ions) - 2 + 3]
        # membrane potential for neuron in previous time step
        phi_NE_ = phi_N_ - phi_E_

        # Assign states
        assert(len(ss) == self.N_states)
        m_NaT, h_NaT, m_NaP, h_NaP, m_KDR, m_KA, h_KA, y, D1, D2 = ss

        # Initial return arguments
        F_expressions = [ufl.zero()]*self.N_states

        # get rate functions
        alpha_m_NaT = self.alpha_m_NaT(phi_NE_) # persistent sodium (NaP)
        beta_m_NaT = self.beta_m_NaT(phi_NE_)   # persistent sodium (NaP)
        alpha_h_NaT = self.alpha_h_NaT(phi_NE_) # persistent sodium (NaP)
        beta_h_NaT = self.beta_h_NaT(phi_NE_)   # persistent sodium (NaP)
        alpha_m_NaP = self.alpha_m_NaP(phi_NE_) # persistent sodium (NaP)
        beta_m_NaP = self.beta_m_NaP(phi_NE_)   # persistent sodium (NaP)
        alpha_h_NaP = self.alpha_h_NaP(phi_NE_) # persistent sodium (NaP)
        beta_h_NaP = self.beta_h_NaP(phi_NE_)   # persistent sodium (NaP)
        alpha_m_KDR = self.alpha_m_KDR(phi_NE_) # potassium delayed rectifier (KDR)
        beta_m_KDR = self.beta_m_KDR(phi_NE_)   # potassium delayed rectifier (KDR)
        alpha_m_KA = self.alpha_m_KA(phi_NE_)   # transient potassium (KA)
        beta_m_KA = self.beta_m_KA(phi_NE_)     # transient potassium (KA)
        alpha_h_KA = self.alpha_h_KA(phi_NE_)   # transient potassium (KA)
        beta_h_KA = self.beta_h_KA(phi_NE_)     # transient potassium (KA)
        F_Glu = self.F_Glu(Glu_E_)

        # Expressions
        F_expressions[0] = alpha_m_NaT*(1.0 - m_NaT) - beta_m_NaT*m_NaT
        F_expressions[1] = alpha_h_NaT*(1.0 - h_NaT) - beta_h_NaT*h_NaT
        F_expressions[2] = alpha_m_NaP*(1.0 - m_NaP) - beta_m_NaP*m_NaP
        F_expressions[3] = alpha_h_NaP*(1.0 - h_NaP) - beta_h_NaP*h_NaP
        F_expressions[4] = alpha_m_KDR*(1.0 - m_KDR) - beta_m_KDR*m_KDR
        F_expressions[5] = alpha_m_KA*(1.0 - m_KA) - beta_m_KA*m_KA
        F_expressions[6] = alpha_h_KA*(1.0 - h_KA) - beta_h_KA*h_KA
        F_expressions[7] = k2*D1 - k1*F_Glu*y
        F_expressions[8] = k1*F_Glu*y + k4*D2 - (k2 + k3)*D1
        F_expressions[9] = k3*D1 - k4*D2

        # Return results
        return as_vector(F_expressions)
