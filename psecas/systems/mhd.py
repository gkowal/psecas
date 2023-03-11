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
            P                - the Prandtl number (default 0)
            beta             - the plasma-beta parameter (default 1)
            zeta             - the balance between thermal and magnetic pressure to guarantee
                               uniform total pressure; 0 means the total pressure is completely
                               balanced by thermal pressure, 1 by the y magnetic field component
                               (default 0)
            a                - the thickness of the current sheet (default 1)
            Bshear           - the strength of the sheared magnetic field component
                               (in the X direction, default 1)
            Bguide           - the guide field (in the Y direction, default 0)
            Vshear           - the amplitude of the velocity shear along the X direction
                               (default 0)
            Vangle           - the angle in radians between the direction of the velocity
                               shear and the X axis in the XY plane (default 0)
            problem          - the equilibrium configuration for the selected problem
                               (only 'tearing' implemented so far)
            vector_potential - use the magnetic vector potential to express the magnetic field
                               (default 'False')
    """
    def __init__(self, grid, kx, theta=0, z1=-0.5, z2=0.5, a=1, adiabatic_index=5/3, \
                    S=1e4, P=0, beta=1, zeta=0, Bshear=1, Bguide=0, Vshear=0, Vangle=0, \
                    problem='tearing', periodic=True, vector_potential=False):
        import numpy as np

        self.problem  = problem
        self.periodic = periodic

        self.__S      = S
        self.__P      = P
        self.__a      = a
        self.__theta  = theta
        self.__gamma  = adiabatic_index
        self.__beta   = beta
        self.__Bshear = Bshear
        self.__Bguide = Bguide
        self.__Vshear = Vshear
        self.__Vangle = Vangle

        if S > 0:
            self.eta       = 1/S
            self.resistive = True
        else:
            self.eta       = 0
            self.resistive = False

        if P > 0 and S > 0:
            self.nu = P/S
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
            self.gm        = 1
            self.gmm1      = 0

        self.csnd2 = self.__gamma * self.__beta * self.__Bshear**2 / 2

        if abs(Vshear) > 0:
            self.shear = True
        else:
            self.shear = False

        self.kv = kx
        self.kx = kx * np.cos(theta)
        self.ky = kx * np.sin(theta)
        self.z1 = z1
        self.z2 = z2

        self.zeta = zeta

        self.vector_potential = vector_potential

        self.grid = grid
        self.grid.bind_to(self.make_background)


        # Create initial background
        self.make_background()

        # Variables to solve for
        self.variables = [ "ddn"]
        self.labels    = [ r"$\delta \rho$" ]
        if self.adiabatic:
            self.variables.append("dpr")
            self.labels.append(r"$\delta p$")
        self.variables.append("dvx")
        self.variables.append("dvy")
        self.variables.append("dvz")
        self.labels.append(r"$\delta v_x$")
        self.labels.append(r"$\delta v_y$")
        self.labels.append(r"$\delta v_z$")
        if vector_potential:
            self.variables.append("dAx")
            self.variables.append("dAy")
            self.variables.append("dAz")
            self.labels.append(r"$\delta A_x$")
            self.labels.append(r"$\delta A_y$")
            self.labels.append(r"$\delta A_z$")
        else:
            self.variables.append("dBx")
            self.variables.append("dBy")
            self.variables.append("dBz")
            self.labels.append(r"$\delta B_x$")
            self.labels.append(r"$\delta B_y$")
            self.labels.append(r"$\delta B_z$")

        # Number of equations in system
        self.dim = len(self.variables)

        # Boundary conditions
        if self.periodic:
            self.boundaries = [ False ]*self.dim
        else:
            self.boundaries = [ True  ]*self.dim
            self.extra_binfo = [ [ 'Dirichlet', 'Dirichlet' ] ]*self.dim

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Equations (Careful! No space behind minus)
        eq1 = "sigma*ddn    = -dDndz*dvz" \
                               +" -Dn*(1j*kx*dvx +1j*ky*dvy +dz(dvz))"
        if self.adiabatic:
            eq2 = "sigma*dpr    = -dPrdz*dvz" \
                            +" -gm*Pr*(1j*kx*dvx +1j*ky*dvy +dz(dvz))"
            eq3 = "sigma*dvx*Dn = -1j*kx*dpr"
            eq4 = "sigma*dvy*Dn = -1j*ky*dpr"
            eq5 = "sigma*dvz*Dn = -dz(dpr)"
        else:
            eq3 = "sigma*dvx*Dn = -1j*kx*csnd2*ddn"
            eq4 = "sigma*dvy*Dn = -1j*ky*csnd2*ddn"
            eq5 = "sigma*dvz*Dn = -csnd2*dz(ddn)"

        if vector_potential:
            eq3 += " +dBxdz*(1j*kx*dAy -1j*ky*dAx)" \
                +" -By*(kx**2*dAz +ky**2*dAz +1j*kx*dz(dAx) +1j*ky*dz(dAy))"
            eq4 += " +dBydz*(1j*kx*dAy -1j*ky*dAx)" \
                +" +Bx*(kx**2*dAz +ky**2*dAz +1j*kx*dz(dAx) +1j*ky*dz(dAy))"
            eq5 += " -dBxdz*(1j*ky*dAz -dz(dAy)) +dBydz*(1j*kx*dAz -dz(dAx))" \
                +" -Bx*(+kx**2*dAy -kx*ky*dAx +1j*ky*dz(dAz) -dz(dz(dAy)))" \
                +" +By*(+ky**2*dAx -kx*ky*dAy +1j*kx*dz(dAz) -dz(dz(dAx)))"
            eq6 = "sigma*dAx = -By*dvz"
            eq7 = "sigma*dAy = +Bx*dvz"
            eq8 = "sigma*dAz = -Bx*dvy +By*dvx"
        else:
            eq3 += " +dBxdz*dBz -1j*By*(kx*dBy -ky*dBx)"
            eq4 += " +dBydz*dBz +1j*Bx*(kx*dBy -ky*dBx)"
            eq5 += " -dBxdz*dBx -dBydz*dBy" \
                +" +Bx*(1j*kx*dBz -dz(dBx))" \
                +" +By*(1j*ky*dBz -dz(dBy))"
            eq6 = "sigma*dBx = -Bx*(1j*ky*dvy +dz(dvz)) -dBxdz*dvz +1j*ky*By*dvx"
            eq7 = "sigma*dBy = -By*(1j*kx*dvx +dz(dvz)) -dBydz*dvz +1j*kx*Bx*dvy"
            eq8 = "sigma*dBz = +1j*kx*Bx*dvz +1j*ky*By*dvz"

        if self.shear:
            eq1 += " -1j*kx*Vx*ddn -1j*ky*Vy*ddn"
            if self.adiabatic:
                eq2 += " -1j*kx*Vx*dpr -1j*ky*Vy*dpr"
            eq3 += " -1j*kx*Dn*Vx*dvx -1j*ky*Dn*Vy*dvx -Dn*dVxdz*dvz"
            eq4 += " -1j*kx*Dn*Vx*dvy -1j*ky*Dn*Vy*dvy -Dn*dVydz*dvz"
            eq5 += " -1j*kx*Dn*Vx*dvz -1j*ky*Dn*Vy*dvz"

            if vector_potential:
                eq6 += " +Vy*(1j*kx*dAy -1j*ky*dAx)"
                eq7 += " -Vx*(1j*kx*dAy -1j*ky*dAx)"
                eq8 += " -Vx*(1j*kx*dAz -dz(dAz)) -Vy*(1j*ky*dAz -dz(dAy))"
            else:
                eq6 += " +Vx*(1j*ky*dBy +dz(dBz)) +dVxdz*dBz -1j*ky*Vy*dBx"
                eq7 += " +Vy*(1j*kx*dBx +dz(dBz)) +dVydz*dBz -1j*kx*Vx*dBy"
                eq8 += " -1j*kx*Vx*dBz -1j*ky*Vy*dBz"

        if self.resistive:
            if vector_potential:
                if self.adiabatic:
                    eq2 += " +2*gmm1*eta*(dBxdz*(kx**2*dAy +ky**2*dAy -dz(dz(dAy)))" \
                                      +" -dBydz*(kx**2*dAx +ky**2*dAx -dz(dz(dAx))))"
                eq6 += " +eta*(-kx**2*dAx -ky**2*dAx +dz(dz(dAx)))"
                eq7 += " +eta*(-kx**2*dAy -ky**2*dAy +dz(dz(dAy)))"
                eq8 += " +eta*(-kx**2*dAz -ky**2*dAz +dz(dz(dAz)))"
            else:
                if self.adiabatic:
                    eq2 += " -2*gmm1*eta*(dBxdz*(1j*kx*dBz -dz(dBx))" \
                                      +" +dBydz*(1j*ky*dBz -dz(dBy)))"
                eq6 += " +eta*(-kx**2*dBx -ky**2*dBx +dz(dz(dBx)))"
                eq7 += " +eta*(-kx**2*dBy -ky**2*dBy +dz(dz(dBy)))"
                eq8 += " +eta*(-kx**2*dBz -ky**2*dBz +dz(dz(dBz)))"

        if self.viscous:
            eq3 += " +nu*(dDndz*(1j*kx*dvz +dz(dvx))" \
                +" +Dn*((-4*kx**2*dvx -kx*ky*dvy +1j*kx*dz(dvz))/3 -ky**2*dvx +dz(dz(dvx))))"
            eq4 += " +nu*(dDndz*(1j*ky*dvz +dz(dvy))" \
                +" +Dn*((-4*ky**2*dvy -kx*ky*dvx +1j*ky*dz(dvz))/3 -kx**2*dvy +dz(dz(dvy))))"
            eq5 += " +nu*(dDndz*(-2j*kx*dvx -2j*ky*dvy +4*dz(dvz))/3" \
                +" +Dn*((4*dz(dz(dvz)) +1j*kx*dz(dvx) +1j*ky*dz(dvy))/3 -kx**2*dvz -ky**2*dvz))"
            if self.shear:
                if self.adiabatic:
                    eq2 += " +gmm1*nu*((Vx*dVxdz +Vy*dVydz)*dz(ddn) +(Vx*d2Vxdz + Vy*d2Vydz)*ddn" \
                        +" +dDndz*(dVxdz*dvx +dVydz*dvy) +Dn*(d2Vxdz*dvx +d2Vydz*dvy)" \
                        +" +Vx*(dDndz*(1j*kx*dvz +dz(dvx)) +Dn*(1j*kx*dz(dvz)/3 -kx*ky*dvy/3 -ky**2*dvx +dz(dz(dvx)) -4*kx**2*dvx/3))" \
                        +" +Vy*(dDndz*(1j*ky*dvz +dz(dvy)) +Dn*(1j*ky*dz(dvz)/3 -kx*ky*dvx/3 -ky**2*dvy +dz(dz(dvy)) -4*ky**2*dvy/3)))"
                eq3 += " +nu*(dVxdz*dz(ddn) +d2Vxdz*ddn)"
                eq4 += " +nu*(dVydz*dz(ddn) +d2Vydz*ddn)"
                eq5 += " +nu*(1j*kx*dVxdz +1j*ky*dVydz)*ddn"

        if self.adiabatic:
            self.equations = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]
        else:
            self.equations = [eq1, eq3, eq4, eq5, eq6, eq7, eq8]

    @property
    def S(self):
        return self.__S

    @S.setter
    def S(self, S):
        if S > 0:
            self.resistive = True
            self.__S = S
            self.eta = 1/S
            self.nu  = self.__P/S
        else:
            self.resistive = False
            self.__S = 0
            self.eta = 0
            self.nu  = 0
        self.make_background()

    @property
    def P(self):
        return self.__P

    @P.setter
    def P(self, P):
        if P > 0 and self.__S > 0:
            self.viscous = True
            self.__P = P
            self.nu   = P/self.__S
            self.make_background()
        else:
            self.viscous = True
            self.__P = 0
            self.nu  = 0
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
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, theta):
        import numpy as np

        self.__theta = theta

        self.kx = self.kv * np.cos(theta)
        self.ky = self.kv * np.sin(theta)

    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, beta):
        if beta > 0:
            self.__beta = beta
            self.csnd2  = self.__gamma * self.__beta * self.__Bshear**2 / 2
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
        else:
            self.adiabatic = False
            self.__gamma   = 1
            self.gm        = 1
            self.gmm1      = 0
        self.csnd2     = self.__gamma * self.__beta * self.__Bshear**2 / 2
        self.make_background()

    @property
    def Bshear(self):
        return self.__Bshear

    @Bshear.setter
    def Bshear(self, Bshear):
        self.__Bshear = Bshear
        self.csnd2  = self.__gamma * self.__beta * self.__Bshear**2 / 2
        self.make_background()

    @property
    def Bguide(self):
        return self.__Bguide

    @Bguide.setter
    def Bguide(self, Bguide):
        self.__Bguide = Bguide
        self.make_background()

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
    def Vangle(self):
        return self.__Vangle

    @Vangle.setter
    def Vangle(self, Vangle):
        self.__Vangle = Vangle
        self.make_background()

    def make_background(self):
        import numpy as np
        from sympy import sqrt, tanh, cosh, diff, lambdify, symbols

        z    = symbols("z")

        zg   = self.grid.zg

        z1   = self.z1
        z2   = self.z2

        zeta  = self.zeta
        csnd2 = self.csnd2

        a    = self.__a
        beta = self.__beta
        Bg   = self.__Bguide
        Bs   = self.__Bshear
        Vs   = self.__Vshear * np.cos(self.__Vangle)
        Vg   = self.__Vshear * np.sin(self.__Vangle)

        if self.problem == 'TR' or self.problem == 'tearing':
            pmag = Bs**2 / 2
            if self.periodic:
                if self.adiabatic:
                    Dn_sym = 1
                    Pr_sym = pmag * (beta + (1 - zeta) * (1 / cosh((z - z1) / a)**2 \
                                                        + 1 / cosh((z - z2) / a)**2))
                else:
                    Dn_sym = 1 + pmag / csnd2 * (1 - zeta) * (1 / cosh((z - z1) / a)**2 \
                                                            + 1 / cosh((z - z2) / a)**2)
                Vx_sym = Vs * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                Vy_sym = Vg * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                Bx_sym = Bs * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                By_sym = sqrt(Bg**2 + zeta * Bs**2 * (1 / cosh((z - z1) / a)**2 \
                                                    + 1 / cosh((z - z1) / a)**2))
            else:
                if self.adiabatic:
                    Dn_sym = 1
                    Pr_sym = pmag * (beta + (1 - zeta) / cosh(z / a)**2)
                else:
                    Dn_sym = 1 + pmag / csnd2 * (1 - zeta) / cosh(z / a)**2
                Vx_sym = Vs * tanh(z / a)
                Vy_sym = Vg * tanh(z / a)
                Bx_sym = Bs * tanh(z / a)
                By_sym = sqrt(Bg**2 + zeta * Bs**2 / cosh(z / a)**2)
        elif self.problem == 'KH' or self.problem == 'kelvin-helmholtz':
            pmag = (Bs**2 + Bg**2) / 2
            if self.periodic:
                Dn_sym = 1
                Pr_sym = (1 + beta) * pmag
                Vx_sym = Vs * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                Vy_sym = Vg * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1)
                Bx_sym = Bs
                By_sym = Bg
            else:
                Dn_sym = 1 + tanh(z / a) / 2
                Pr_sym = (1 + beta) * pmag
                Vx_sym = Vs * tanh(z / a)
                Vy_sym = Vg * tanh(z / a)
                Bx_sym = Bs
                By_sym = Bg
        else:
            Dn_sym = 1
            pr_sym = (1 + beta) / 2
            Vx_sym = 0
            Bx_sym = 1
            By_sym = Bg

        dDndz_sym  = diff(Dn_sym, z)
        dVxdz_sym  = diff(Vx_sym, z)
        d2Vxdz_sym = diff(dVxdz_sym, z)
        dVydz_sym  = diff(Vy_sym, z)
        d2Vydz_sym = diff(dVydz_sym, z)
        dBxdz_sym  = diff(Bx_sym, z)
        dBydz_sym  = diff(By_sym, z)
        if self.adiabatic:
            dPrdz_sym  = diff(Pr_sym, z)

        self.Dn     = lambdify(z, Dn_sym)(zg)
        self.dDndz  = lambdify(z, dDndz_sym)(zg)
        self.Vx     = lambdify(z, Vx_sym)(zg)
        self.dVxdz  = lambdify(z, dVxdz_sym)(zg)
        self.d2Vxdz = lambdify(z, d2Vxdz_sym)(zg)
        self.Vy     = lambdify(z, Vy_sym)(zg)
        self.dVydz  = lambdify(z, dVydz_sym)(zg)
        self.d2Vydz = lambdify(z, d2Vydz_sym)(zg)
        self.Bx     = lambdify(z, Bx_sym)(zg)
        self.dBxdz  = lambdify(z, dBxdz_sym)(zg)
        self.By     = lambdify(z, By_sym)(zg)
        self.dBydz  = lambdify(z, dBydz_sym)(zg)
        if self.adiabatic:
            self.Pr     = lambdify(z, Pr_sym)(zg)
            self.dPrdz  = lambdify(z, dPrdz_sym)(zg)
