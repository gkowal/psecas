class MHD:
    """
        The linearized compressible MHD framework with the adiabatic equation of state.
        The example equilibrium is for tearing instability with uniform density, total pressure,
        and electric field (neglecting the dissipative terms).
        Periodic and non-periodic grids are supported.

        Arguments:
            grid             - the grid on which the computation is done
            kx               - the wavenumber
            theta            - the angle in radians between the wavenumber direction and X axis
                               (default 0)
            z1, z2           - the positions of the current sheets (only for the periodic grid,
                               default -0.5 and 0.5, respectively)
            adiabatic_index  - the adiabatic index
            S                - the Lundquist number (default 1e5)
            Pr               - the Prandtl number (default 0)
            beta             - the plasma-beta parameter (default 1)
            a                - the thickness of the current sheet (default 1)
            Bguide           - the guide field (in the Y direction, default 0)
            Vshear           - the amplitude of the velocity shear along the X direction
                               (default 0)
            problem          - the equilibrium configuration for the selected problem
                               (only 'tearing' implemented so far)
            vector_potential - use the magnetic vector potential to express the magnetic field
                               (default 'False')
    """
    def __init__(self, grid, kx, theta=0, z1=-0.5, z2=0.5, a=1, adiabatic_index=5/3, \
                    S=1e4, Pr=0, beta=1, Bguide=0, Vshear=0, \
                    problem='tearing', periodic=True, vector_potential=False):
        import numpy as np

        self.problem  = problem
        self.periodic = periodic

        self.__S      = S
        self.__Pr     = Pr
        self.__a      = a
        self.__gamma  = adiabatic_index
        self.__beta   = beta
        self.__Bguide = Bguide
        self.__Vshear = Vshear

        if S > 0:
            self.eta       = 1/S
            self.resistive = True
        else:
            self.eta       = 0
            self.resistive = False

        if Pr > 0 and S > 0:
            self.nu = Pr/S
            self.viscous = True
        else:
            self.nu = 0
            self.viscous = False

        if adiabatic_index > 1:
            self.adiabatic = True
            self.gm        = self.__gamma
            self.gmm1      = self.__gamma - 1
        else:
            self.adiabatic = False

        if abs(Vshear) > 0:
            self.shear = True
        else:
            self.shear = False

        self.kx  = kx * np.cos(theta)
        self.ky  = kx * np.sin(theta)
        self.z1  = z1
        self.z2  = z2

        self.vector_potential = vector_potential

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Create initial background
        self.make_background()

        # Variables to solve for
        if vector_potential:
            self.variables = ["ddn", "dpr", "dvx", "dvy", "dvz", "dAx", "dAy", "dAz"]
            self.labels = [
                r"$\delta \rho$",
                r"$\delta p$"  ,
                r"$\delta v_x$",
                r"$\delta v_y$",
                r"$\delta v_z$",
                r"$\delta A_x$",
                r"$\delta A_y$",
                r"$\delta A_z$",
            ]
        else:
            self.variables = ["ddn", "dpr", "dvx", "dvy", "dvz", "dBx", "dBy", "dBz"]
            self.labels = [
                r"$\delta \rho$",
                r"$\delta p$"  ,
                r"$\delta v_x$",
                r"$\delta v_y$",
                r"$\delta v_z$",
                r"$\delta B_x$",
                r"$\delta B_y$",
                r"$\delta B_z$",
            ]

        # Boundary conditions
        if self.periodic:
            self.boundaries = [False, False, False, False, False, False, False, False]
        else:
            self.boundaries = [True, True, True, True, True, True, True, True]
            self.extra_binfo = [['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet'],
                                ['Dirichlet', 'Dirichlet']]

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Equations (Careful! No space behind minus)
        eq1 = "sigma*ddn    = -dDndz*dvz" \
                           +"    -Dn*(1j*kx*dvx +1j*ky*dvy +dz(dvz))"
        eq2 = "sigma*dpr    = -dprdz*dvz" \
                           +" -gm*pr*(1j*kx*dvx +1j*ky*dvy +dz(dvz))"
        eq3 = "sigma*dvx*Dn = -1j*kx*dpr"
        eq4 = "sigma*dvy*Dn = -1j*ky*dpr"
        eq5 = "sigma*dvz*Dn = -dz(dpr)"

        if vector_potential:
            eq3 += " +1j*dBxdz*(kx*dAy -ky*dAx)" \
                  +" -By*(kx**2*dAz +ky**2*dAz +1j*kx*dz(dAx) +1j*ky*dz(dAy))"
            eq4 += " +1j*dBydz*(kx*dAy -ky*dAx)" \
                  +" +Bx*(kx**2*dAz +ky**2*dAz +1j*kx*dz(dAx) +1j*ky*dz(dAy))"
            eq5 += " -dBydz*(dz(dAx) -1j*kx*dAz) -dBxdz*(1j*ky*dAz -dz(dAy))" \
                  +" +Bx*(-kx**2*dAy +kx*ky*dAx -1j*ky*dz(dAz) +dz(dz(dAy)))" \
                  +" +By*(-kx*ky*dAy +ky**2*dAx +1j*kx*dz(dAz) -dz(dz(dAx)))"
            eq6 = "sigma*dAx = -By*dvz"
            eq7 = "sigma*dAy = +Bx*dvz"
            eq8 = "sigma*dAz = -Bx*dvy +By*dvx"

        else:
            eq3 += " +dBxdz*dBz -1j*By*(kx*dBy -ky*dBx)"
            eq4 += " +dBydz*dBz +1j*Bx*(kx*dBy -ky*dBx)"
            eq5 += " -dBydz*dBy -dBxdz*dBx" \
                  +" +Bx*(1j*kx*dBz -dz(dBx))" \
                  +" +By*(1j*ky*dBz -dz(dBy))"
            eq6 = "sigma*dBx = +1j*ky*(By*dvx -Bx*dvy)" \
                            +" -dBxdz*dvz -Bx*dz(dvz)"
            eq7 = "sigma*dBy = -1j*kx*(By*dvx -Bx*dvy)" \
                            +" -dBydz*dvz -By*dz(dvz)"
            eq8 = "sigma*dBz = +1j*kx*Bx*dvz +1j*ky*By*dvz"

        if self.viscous:
            eq3 += " +nu*(dDndz*(1j*kx*dvz +dz(dvx))" \
                 +" -4*kx**2*dvx/3  -ky**2*dvy    +dz(dz(dvx))       -kx*ky*dvy/3 +1j*kx*dz(dvz)/3)"
            eq4 += " +nu*(dDndz*(1j*ky*dvz +dz(dvy))" \
                 +"   -kx**2*dvy  -4*ky**2*dvy/3  +dz(dz(dvy))       -kx*ky*dvx/3 +1j*ky*dz(dvz)/3)"
            eq5 += " +nu*(-2*dDndz*(1j*kx*dvx +1j*ky*dvy -2*dz(dvz))/3" \
                 +"   -kx**2*dvz    -ky**2*dvz  +4*dz(dz(dvz))/3 +1j*kx*dz(dvx)/3 +1j*ky*dz(dvy)/3)"

        if self.resistive:
            if vector_potential:
                eq2 += " -2*gmm1*eta*(dBxdz*(-kx**2*dAy +kx*ky*dAx -1j*ky*dz(dAz) +dz(dz(dAy)))" \
                                  +" +dBydz*(-kx*ky*dAy +ky**2*dAx +1j*kx*dz(dAz) -dz(dz(dAx))))"
                eq6 += " +eta*(-kx**2*dAx -ky**2*dAx +dz(dz(dAx)))"
                eq7 += " +eta*(-kx**2*dAy -ky**2*dAy +dz(dz(dAy)))"
                eq8 += " +eta*(-kx**2*dAz -ky**2*dAz +dz(dz(dAz)))"
            else:
                eq2 += " -2*gmm1*eta*(dBxdz*(1j*kx*dBz -dz(dBx))" \
                                  +" +dBydz*(1j*ky*dBz -dz(dBy)))"
                eq6 += " +eta*(-kx**2*dBx -ky**2*dBx +dz(dz(dBx)))"
                eq7 += " +eta*(-kx**2*dBy -ky**2*dBy +dz(dz(dBy)))"
                eq8 += " +eta*(-kx**2*dBz -ky**2*dBz +dz(dz(dBz)))"

        if self.shear:
            eq1 += " -1j*kx*Vx*ddn"
            eq2 += " -1j*kx*Vx*dpr"
            eq3 += " -1j*kx*Dn*Vx*dvx -Dn*dVxdz*dvx"
            eq4 += " -1j*kx*Dn*Vx*dvy"
            eq5 += " -1j*kx*Dn*Vx*dvz"
            if vector_potential:
                eq7 += " -1j*kx*Vx*dAy"
                eq8 += " +Vx*(-1j*kx*dAz +dz(dAz))"
            else:
                eq6 += " -1j*ky*Vx*dBy -dVxdz*dBz -Vx*dz(dBz)"
                eq7 += " +1j*kx*Vx*dBy"
                eq8 += " +1j*kx*Vx*dBz"

        self.equations = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

    @property
    def S(self):
        return self.__S

    @S.setter
    def S(self, S):
        if S > 0:
            self.resistive = True
            self.__S = S
            self.eta = 1/S
            self.nu  = self.__Pr/S
        else:
            self.resistive = False
            self.__S = 0
            self.eta = 0
            self.nu  = 0
        self.make_background()

    @property
    def Pr(self):
        return self.__Pr

    @Pr.setter
    def Pr(self, Pr):
        if Pr > 0 and self.__S > 0:
            self.viscous = True
            self.__Pr = Pr
            self.nu   = Pr/self.__S
            self.make_background()
        else:
            self.viscous = True
            self.__Pr = 0
            self.nu   = 0
            self.make_background()

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, a):
        if a > 0:
                self.__a = a
                self.make_background()
        else:
            print('a must be > 0! Current value a={} is unchanged'.format(self.__a))

    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, beta):
        if beta > 0:
            self.__beta = beta
            self.make_background()
        else:
            print('beta must be > 0! Current value beta={} is unchanged'.format(self.__beta))

    @property
    def adiabatic_index(self):
        return self.__gamma

    @adiabatic_index.setter
    def adiabatic_index(self, gamma):
        if gamma > 1:
            self.adiabatic = True
            self.__gamma   = gamma
            self.gm        = gamma
            self.gmm1      = gamma - 1
            self.make_background()
        else:
            self.adiabatic = False
            self.__gamma   = 1
            self.gm        = 1
            self.gmm1      = 0

    @property
    def Vshear(self):
        return self.__Vshear

    @Vshear.setter
    def Vshear(self, Vshear):
        if abs(Vshear) > 0:
            self.shear    = True
            self.__Vshear = Vshear
        else:
            self.shear    = False
            self.__Vshear = 0
        self.make_background()

    @property
    def Bguide(self):
        return self.__Bguide

    @Bguide.setter
    def Bguide(self, Bguide):
        self.__Bguide = Bguide
        self.make_background()

    def make_background(self):
        from sympy import tanh, diff, lambdify, symbols

        z    = symbols("z")

        zg   = self.grid.zg

        z1   = self.z1
        z2   = self.z2

        beta = self.__beta
        Bg   = self.__Bguide

        if self.problem == 'tearing':
            a  = self.__a
            Vs = self.__Vshear

            if self.periodic:
                Dn_sym = 1
                pr_sym = (1 + beta) / 2 - (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)**2 / 2
                Vx_sym = Vs * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                Bx_sym =      (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                By_sym = Bg
            else:
                Dn_sym = 1
                pr_sym = (1 + beta) / 2 - tanh(z / a)**2 / 2
                Vx_sym = Vs * tanh(z / a)
                Bx_sym =      tanh(z / a)
                By_sym = Bg
        else:
            Dn_sym = 1
            pr_sym = (1 + beta) / 2
            Vx_sym = 0
            Bx_sym = 1
            By_sym = Bg

        dDndz_sym  = diff(Dn_sym, z)
        d2Dndz_sym = diff(dDndz_sym, z)
        dprdz_sym  = diff(pr_sym, z)
        d2prdz_sym = diff(dprdz_sym, z)
        dVxdz_sym  = diff(Vx_sym, z)
        d2Vxdz_sym = diff(dVxdz_sym, z)
        dBxdz_sym  = diff(Bx_sym, z)
        d2Bxdz_sym = diff(dBxdz_sym, z)
        dBydz_sym  = diff(By_sym, z)
        d2Bydz_sym = diff(dBydz_sym, z)

        self.Dn     = lambdify(z, Dn_sym)(zg)
        self.dDndz  = lambdify(z, dDndz_sym)(zg)
        self.d2Dndz = lambdify(z, d2Dndz_sym)(zg)
        self.pr     = lambdify(z, pr_sym)(zg)
        self.dprdz  = lambdify(z, dprdz_sym)(zg)
        self.d2prdz = lambdify(z, d2prdz_sym)(zg)
        self.Vx     = lambdify(z, Vx_sym)(zg)
        self.dVxdz  = lambdify(z, dVxdz_sym)(zg)
        self.d2Vxdz = lambdify(z, d2Vxdz_sym)(zg)
        self.Bx     = lambdify(z, Bx_sym)(zg)
        self.dBxdz  = lambdify(z, dBxdz_sym)(zg)
        self.d2Bxdz = lambdify(z, d2Bxdz_sym)(zg)
        self.By     = lambdify(z, By_sym)(zg)
        self.dBydz  = lambdify(z, dBydz_sym)(zg)
        self.d2Bydz = lambdify(z, d2Bydz_sym)(zg)
