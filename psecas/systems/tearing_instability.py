class TearingClassicalMHD:
    """
        The class implements a linear stability analysis of the tearing instability
        within a framework of linearized, incompressible magnetohydrodynamics (MHD),
        incorporating both viscous and resistive effects. These effects are quantified
        by the Prandtl and Lundquist numbers, respectively.

        In the two-dimensional scenario, where either kx = 0 or ky = 0, the analysis
        revolves around two linearized equations that govern the evolution of
        the Z components of velocity and magnetic field. This setup provides insights
        into the dynamics of these components under specified wave vector conditions.

        For the three-dimensional case, the analysis expands to include four
        linearized equations. These equations are used to resolve the perturbations
        in both the Y and Z components of the velocity and magnetic field. This
        comprehensive approach allows for a detailed examination of the behavior of
        these fields in a three-dimensional space, addressing more complex interactions
        and disturbances.

        The assumed equilibrium state varies solely along the Z coordinate,
        characterized by specific profiles:
            - The X component of the equilibrium magnetic field is defined by
                Bx(z) = ½ [tanh(z + w) + tanh(z - w)],
            - The X component of the equilibrium velocity field is
                Ux(z) = ½ [tanh(z + w) - tanh(z - w)],
            - The Y component of the equilibrium magnetic field is
                By(z) = ½ [sech(z + w) + sech(z - w)] + Bguide,
              where Bguide denotes the strength of the guide field,
            - The Y component of the equilibrium velocity field is
                Uy(z) = ½ [sech(z + w) - sech(z - w)].

        Additionally, the class supports both periodic and non-periodic grids,
        accommodating various boundary conditions and geometrical configurations.

        This flexibility allows for a comprehensive analysis of magnetic field
        dynamics under different physical constraints.
    """
    def __init__(self, grid, kx=0, ky=0, z1=-0.5, z2=0.5, a=1, w=0, \
                S=1e4, Pr=0, ζ=0, ξ=0, ϵ=0, Bguide=0, kh=None, shear=True, periodic=True):
        import numpy as np

        # Validation checks
        if kx < 0:
            raise ValueError("kx must be >= 0")
        if a <= 0:
            raise ValueError("a must be > 0")
        if w < 0:
            raise ValueError("w must be >= 0")
        if S <= 0:
            raise ValueError("S must be > 0")
        if Pr < 0:
            raise ValueError("Pr must be >= 0")
        if ζ < 0:
            raise ValueError("ζ must be >= 0")
        if ϵ < 0:
            raise ValueError("ϵ must be >= 0")
        if kh is not None and kh <= 0:
            raise ValueError("kh must be > 0")

        self.periodic = periodic
        self.shear    = shear

        self.__a  = a
        self.__w  = w
        self.__S  = S
        self.__Pr = Pr
        self.__Bg = Bguide
        self.__ζ  = ζ

        self.δ    = a
        self.η    = 1/S
        self.ν    = Pr/S
        self.ξ    = ξ
        self.ϵ    = ϵ
        if kh is not None:
            self.κ = κ = 1/S/kh**2

        self.kx   = kx
        self.ky   = ky
        self.z1   = z1
        self.z2   = z2
        self.λ    = kx*a

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Create initial background
        self.make_background()

        Hall = ϵ > 0

        # Linearized equations for three cases: 1) ky = 0, 2) kx = 0, 3) kx != 0 and ky != 0
        if Hall:
            if np.isclose(ky, 0.0):
                # Equations (Careful! No space behind minus)
                wy_lhs = "sigma*(dz(duz,2) -kx**2*duz)"
                bz_lhs = "sigma*dbz"
                wz_lhs = "sigma*duy"
                by_lhs = "sigma*dby"

                wy_rhs_adv = ""
                wy_rhs_pre = ""
                wy_rhs_lor = " +1j*kx*(Bx*(dz(dbz,2) -kx**2*dbz) -d2Bxdz*dbz)"
                wy_rhs_vis = ""
                wz_rhs_adv = ""
                wz_rhs_pre = ""
                wz_rhs_lor = " +1j*kx*Bx*dby +dBydz*dbz"
                wz_rhs_vis = ""

                by_rhs_ind = " +1j*kx*Bx*duy -dBydz*duz"
                by_rhs_res = " +η*(dz(dby,2) -kx**2*dby)"
                by_rhs_hal = " -ϵ*(1j*ξ*dz(dbz,3) -kx*(1j*ξ*kx*dz(dbz) -kx**2*Bx*dbz +Bx*dz(dbz,2) -dbz*d2Bxdz))/kx"
                bz_rhs_ind = " +1j*kx*Bx*duz -1j*kx*Ux*dbz +ξ*dz(duz)"
                bz_rhs_res = " +η*(dz(dbz,2) -kx**2*dbz)"
                bz_rhs_hal = " +ϵ*kx*(kx*Bx*dby -1j*dBydz*dbz) -1j*ξ*kx*ϵ*dz(dby)"

                if w > 0 and shear:
                    wy_rhs_adv += " -1j*kx*(Ux*(dz(duz,2) -kx**2*duz) -duz*d2Uxdz)"
                    wz_rhs_adv += " -1j*kx*Ux*duy -dUydz*duz"
                    by_rhs_ind += " -1j*kx*Ux*dby +dUydz*dbz"
                    bz_rhs_ind += " -1j*kx*Ux*dbz"

                if ξ > 0:
                    wy_rhs_lor += " +ξ*(dz(dbz,3) -kx**2*dz(dbz))"
                    wz_rhs_lor += " +ξ*dz(dby)"
                    by_rhs_ind += " +ξ*dz(duy)"
                    bz_rhs_ind += " +ξ*dz(duz)"
                    bz_rhs_hal += " -1j*ϵ*ξ*kx*dz(dby)"

                if Pr > 0:
                    wy_rhs_vis += " +ν*(dz(duz,4) +kx**2*(kx**2*duz -2*dz(duz,2)))"
                    wz_rhs_vis += "+ ν*(dz(duy,2) -kx**2*duy)"

                wy_eq = f"{wy_lhs} ={wy_rhs_adv}{wy_rhs_pre}{wy_rhs_lor}{wy_rhs_vis}"
                wz_eq = f"{wz_lhs} ={wz_rhs_adv}{wz_rhs_pre}{wz_rhs_lor}{wz_rhs_vis}"
                by_eq = f"{by_lhs} ={by_rhs_ind}{by_rhs_res}{by_rhs_hal}"
                bz_eq = f"{bz_lhs} ={bz_rhs_ind}{bz_rhs_res}{bz_rhs_hal}"

                self.variables = ["duy", "duz", "dby", "dbz"]
                self.labels = [
                    r"$\delta u_y$",
                    r"$\delta u_z$",
                    r"$\delta B_y$",
                    r"$\delta B_z$",
                ]
                self.equations = [wz_eq, wy_eq, by_eq, bz_eq]
    
            elif np.isclose(kx, 0.0):
                raise NotImplementedError("Linearized Classical MHD equations for kx = 0 and ky != 0 are currently work in progress.")
            
            else:
                raise NotImplementedError("Linearized Classical MHD equations for kx != 0 and ky != 0 are currently work in progress.")
        else:
            if np.isclose(ky, 0.0):
                # Equations (Careful! No space behind minus)
                wy_lhs = "sigma*(dz(duz,2) -kx**2*duz)"
                bz_lhs = "sigma*dbz"

                wy_rhs_adv = ""
                wy_rhs_pre = ""
                wy_rhs_lor = " +1j*kx*(Bx*(dz(dbz,2) -kx**2*dbz) -d2Bxdz*dbz)"
                wy_rhs_vis = ""
                bz_rhs_ind = " +1j*kx*Bx*duz"
                bz_rhs_res = " +η*(dz(dbz,2) -kx**2*dbz)"

                if w > 0 and shear:
                    wy_rhs_adv += " -1j*kx*(Ux*(dz(duz,2) -kx**2*duz) -d2Uxdz*duz)"
                    bz_rhs_ind += " -1j*kx*Ux*dbz"

                if ξ > 0:
                    wy_rhs_lor += " +ξ*(dz(dbz,3) -kx**2*dz(dbz))"
                    bz_rhs_ind += " +ξ*dz(duz)"

                if Pr > 0:
                    wy_rhs_vis += " +ν*(dz(duz,4) +kx**2*(kx**2*duz -2*dz(duz,2)))"

                wy_eq = f"{wy_lhs} ={wy_rhs_adv}{wy_rhs_pre}{wy_rhs_lor}{wy_rhs_vis}"
                bz_eq = f"{bz_lhs} ={bz_rhs_ind}{bz_rhs_res}"

                self.variables = ["duz", "dbz"]
                self.labels = [
                    r"$\delta u_z$",
                    r"$\delta B_z$",
                ]
                self.equations = [wy_eq, bz_eq]

            elif np.isclose(kx, 0.0):
                raise NotImplementedError("Linearized Classical MHD equations for kx = 0 and ky != 0 are currently work in progress.")

            else:
                raise NotImplementedError("Linearized Classical MHD equations for kx != 0 and ky != 0 are currently work in progress.")

        # Boundary conditions
        if self.periodic:
            self.boundaries = [ False ]*len(self.variables)
        else:
            self.boundaries = [ True ]*len(self.variables)
            self.extra_binfo = [[f'dz({var}) -λ*{var}=0', f'dz({var}) +λ*{var}=0'] for var in self.variables]

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, a):
        self.__a = a
        self.make_background()

    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, w):
        self.__w = w
        self.make_background()

    @property
    def S(self):
        return self.__S

    @S.setter
    def S(self, S):
        if S > 0:
            self.__S  = S
            self.η = 1/S
            self.ν = self.__Pr/S
        else:
            print('S must be > 0! Current value S={} is unchanged'.format(self.__S))

    @property
    def Pr(self):
        return self.__Pr

    @Pr.setter
    def Pr(self, Pr):
        if Pr >= 0:
            self.__Pr = Pr
            self.ν    = self.__Pr/self.__S
        else:
            print('Pr must be >= 0! Current value Pr={} is unchanged'.format(self.__Pr))

    @property
    def Bguide(self):
        return self.__Bg

    @Bguide.setter
    def Bguide(self, Bguide):
        self.__Bg = Bguide
        self.make_background()

    def make_background(self):
        from sympy import symbols, lambdify, diff, tanh, sech

        def sech_stable(x):
            """
            Stable hyperbolic secant for real x.
            Uses: sech(x) = 2*e^{-|x|} / (1 + e^{-2|x|})
            This avoids overflow for large |x| and loss of precision near 0.
            """
            import numpy as np

            x = np.asarray(x, dtype=np.float64)
            t = np.exp(-np.abs(x))          # in [0, 1]
            return (2.0 * t) / (1.0 + t*t)  # even function

        z  = symbols("z")
        zg = self.grid.zg

        kx = self.kx
        ky = self.ky
        a  = self.a
        w  = self.__w
        z1 = self.z1
        z2 = self.z2
        Bg = self.__Bg
        ζ  = self.__ζ

        if self.periodic:
            Bx_sym = (tanh((z - z1 + w) / a) + tanh((z - z1 - w) / a)) / 2 \
                   - (tanh((z - z2 + w) / a) + tanh((z - z2 - w) / a)) / 2 - 1
            By_sym = (sech((z - z1 + w) / a) + sech((z - z1 - w) / a)) / 2 \
                   - (sech((z - z2 + w) / a) + sech((z - z2 - w) / a)) / 2 - 1 \
                   + Bg
            Ux_sym = (tanh((z - z1 + w) / a) - tanh((z - z1 - w) / a)) / 2 \
                   - (tanh((z - z2 + w) / a) - tanh((z - z2 - w) / a)) / 2
            Uy_sym = diff(By_sym, z)
        else:
            Bx_sym =     (tanh((z + w) / a) + tanh((z - w) / a)) / 2
            By_sym = ζ * (sech((z + w) / a) + sech((z - w) / a)) / 2 + Bg
            Ux_sym =     (tanh((z + w) / a) - tanh((z - w) / a)) / 2
            Uy_sym = ζ * (sech((z + w) / a) - sech((z - w) / a)) / 2

        dBxdz_sym  = diff(Bx_sym, z)
        dBydz_sym  = diff(By_sym, z)
        d2Bxdz_sym = diff(dBxdz_sym, z)
        d2Bydz_sym = diff(dBydz_sym, z)

        dUxdz_sym  = diff(Ux_sym   , z)
        dUydz_sym  = diff(Uy_sym   , z)
        d2Uxdz_sym = diff(dUxdz_sym, z)
        d2Uydz_sym = diff(dUydz_sym, z)

        modules = [{"sech": sech_stable}, "numpy"]

        self.Bx     = lambdify(z, Bx_sym    , modules)(zg)
        self.By     = lambdify(z, By_sym    , modules)(zg)
        self.dBxdz  = lambdify(z, dBxdz_sym , modules)(zg)
        self.dBydz  = lambdify(z, dBydz_sym , modules)(zg)
        self.d2Bxdz = lambdify(z, d2Bxdz_sym, modules)(zg)
        self.d2Bydz = lambdify(z, d2Bydz_sym, modules)(zg)

        self.Ux     = lambdify(z, Ux_sym    , modules)(zg)
        self.Uy     = lambdify(z, Uy_sym    , modules)(zg)
        self.dUxdz  = lambdify(z, dUxdz_sym , modules)(zg)
        self.dUydz  = lambdify(z, dUydz_sym , modules)(zg)
        self.d2Uxdz = lambdify(z, d2Uxdz_sym, modules)(zg)
        self.d2Uydz = lambdify(z, d2Uydz_sym, modules)(zg)


class TearingGyrotropicMHD:
    """
        The class implements a linear stability analysis of the tearing instability
        within a framework of linearized, incompressible gyrotropic
        magnetohydrodynamics (MHD), incorporating both viscous and resistive effects.
        These effects are quantified by the Prandtl and Lundquist numbers, respectively.

        The assumed equilibrium state varies solely along the Z coordinate,
        characterized by specific profiles:
            - The X component of the equilibrium magnetic field is defined by
                Bx(z) = tanh(z/a)
            - The Y component of the equilibrium magnetic field is
                By(z) = sech(z/a) + Bguide,
              where Bguide denotes the strength of the guide field,
    """
    def __init__(self, grid, kx=0, ky=0, z1=-0.5, z2=0.5, a=1, \
                    S=1e4, Pr=0, β=0, Δβ=0, ɣpar=3, ɣper=2, ϵ=0, σ=0, periodic=True,
                    normalized=False):
        import numpy as np

        # Validation checks
        if kx <= 0:
            raise ValueError("kx must be >= 0")
        if a  <= 0:
            raise ValueError("a must be > 0")
        if S  <= 0:
            raise ValueError("S must be > 0")
        if Pr <  0:
            raise ValueError("Pr must be >= 0")
        if β  <  0:
            raise ValueError("β must be >= 0")
        if ɣpar <= 0:
            raise ValueError("ɣpar must be > 0")
        if ɣper <= 0:
            raise ValueError("ɣper must be > 0")
        if ϵ < 0:
            raise ValueError("ϵ must be >= 0")

        self.periodic   = periodic
        self.isothermal = np.isclose(ɣpar, 1.0) and np.isclose(ɣper, 1.0)
        self.Hall       = ϵ > 0.0

        self.__a  = a
        self.__S  = S
        self.__Pr = Pr
        self.__β  = β
        self.__Δβ = Δβ

        self.kx   = kx
        self.ky   = ky

        self.δ    = a
        self.z1   = z1
        self.z2   = z2

        self.η    = 1/S
        self.ν    = Pr/S
        self.β0   = β
        self.Δβ0  = Δβ
        self.Δβh  = Δβ / 2
        self.Γ1   = ɣpar + ɣper - 2
        self.Γ2   = ɣpar - 1
        self.Γβ   = 0.5 * (self.Γ1 * self.β0 + self.Γ2 * self.Δβ0)
        self.sqrtG0 = np.sqrt(self.Γβ)
        self.normalized = normalized
        self.ϵ    = ϵ
        self.A    = 1.0 - 0.5 * Δβ
        self.R0   = 1.0 + 0.5 * ((ɣpar + ɣper - 2) * β + ɣpar * Δβ)
        self.χ    = (σ / (kx * a))**2
        self.λ    = kx * a * np.sqrt((self.χ + self.A) / (self.χ + self.R0))

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Create initial background
        self.make_background()

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Pressure-anisotropy variable.
        #
        # normalized=False uses δΔp directly. normalized=True uses the
        # rescaled δπ = δΔp / sqrt(Γβ), and divides the δΔp equation through
        # by sqrt(Γβ), which makes the coupling between the anisotropy and
        # the momentum equations symmetric: both directions then carry a
        # factor sqrt(Γβ) instead of one carrying Γβ and the other 1. Note
        # Γβ/sqrt(Γβ) = sqrt(Γβ), which is why the source term picks up
        # sqrtG0 rather than losing a factor.
        #
        # It is a change of variables, not a change of physics: the two give
        # the same eigenvalues, and tests/test_tearing_normalization.py pins
        # that.
        dp  = "dpi" if normalized else "ddp"
        g0  = "sqrtG0*" if normalized else ""
        pre = "sqrtG0" if normalized else "Γβ"
        div = "/sqrtG0" if normalized else ""

        # Linearized equations for three cases: 1) ky = 0, 2) kx = 0, 3) kx != 0 and ky != 0
        if np.isclose(ky, 0.0):
            wy_lhs = "sigma*(dz(duz,2) -kx**2*duz)"
            wz_lhs = "sigma*duy"
            by_lhs = "sigma*dby"
            bz_lhs = "sigma*dbz"
            dp_lhs = f"sigma*kx*{dp}"

            wy_rhs_adv = ""
            wz_rhs_adv = ""
            wy_rhs_pre = ""
            wz_rhs_pre = ""
            wy_rhs_lor = " +1j*kx*Bx*((dz(dbz,2) -kx**2*dbz) +2*By**2/δ**2*dbz)"
            wz_rhs_lor = " +1j*kx*Bx*dby -Bx*By/δ*dbz"
            wy_rhs_vis = ""
            wz_rhs_vis = ""
            wy_rhs_ani = ""
            wz_rhs_ani = ""
            if Pr > 0:
                wy_rhs_vis += " +ν*(dz(duz,4) -2*kx**2*dz(duz,2) +kx**4*duz)"
                wz_rhs_vis += " +ν*(dz(duy,2) -kx**2*duy)"
            if not self.isothermal:
                wy_rhs_ani += f" -kx**2*Bx*{g0}(Bx*dz({dp}) +2*By**2/δ*{dp})"
                wz_rhs_ani += f" -1j*kx*Bx*By*{g0}{dp}"
            if not np.isclose(self.Δβh, 0.0):
                wy_rhs_ani += " -Δβh*kx*Bx*(1j*(1 -2*Bx**2)*dz(dbz,2)" \
                            + " -6j*Bx*By**2/δ*dz(dbz) +1j*(2*By**2/δ**2 -kx**2)*dbz" \
                            + " -2*kx*By*(Bx*dz(dby) +(2*By**2 -Bx**2)/δ*dby))"
                wz_rhs_ani += " -Δβh*(1j*kx*Bx*(1 -2*By**2)*dby -Bx*By/δ*dbz +2*Bx**2*By*dz(dbz))"

            by_rhs_ind = " +1j*kx*Bx*duy +Bx*By/δ*duz"
            bz_rhs_ind = " +1j*kx*Bx*duz"
            by_rhs_res = " +η*(dz(dby,2) -kx**2*dby)"
            bz_rhs_res = " +η*(dz(dbz,2) -kx**2*dbz)"
            by_rhs_hal = ""
            bz_rhs_hal = ""
            if self.Hall:
                by_rhs_hal += " +ϵ*Bx*(dz(dbz,2) -kx**2*dbz +2*By**2/δ**2*dbz)"
                bz_rhs_hal += " +ϵ*kx*Bx*(kx*dby +1j*By/δ*dbz)"

            if not self.isothermal:
                dp_rhs_adv = ""
                dp_rhs_str = f" +{pre}*kx*Bx*(Bx*dz(duz) -1j*kx*By*duy)"
                dp_rhs_res = f" +2*η*Γ2*By/δ*(1j*By*(dz(dbz,2) -kx**2*dbz) -kx*Bx*dz(dby)){div}"
                dp_rhs_vis = ""

            wy_eq = f"{wy_lhs} ={wy_rhs_adv}{wy_rhs_pre}{wy_rhs_lor}{wy_rhs_vis}{wy_rhs_ani}"
            wz_eq = f"{wz_lhs} ={wz_rhs_adv}{wz_rhs_pre}{wz_rhs_lor}{wz_rhs_vis}{wz_rhs_ani}"
            by_eq = f"{by_lhs} ={by_rhs_ind}{by_rhs_res}{by_rhs_hal}"
            bz_eq = f"{bz_lhs} ={bz_rhs_ind}{bz_rhs_res}{bz_rhs_hal}"
            if not self.isothermal:
                dp_eq = f"{dp_lhs} ={dp_rhs_adv}{dp_rhs_str}{dp_rhs_res}{dp_rhs_vis}"

            if self.isothermal:
                self.variables = ["duy", "duz", "dby", "dbz"]
                self.labels = [
                    r"$\delta u_y$",
                    r"$\delta u_z$",
                    r"$\delta B_y$",
                    r"$\delta B_z$",
                ]
                self.equations = [wz_eq, wy_eq, by_eq, bz_eq]
            else:
                self.variables = ["duz", "dbz", "duy", "dby", dp]
                self.labels = [
                    r"$\delta u_z$",
                    r"$\delta B_z$",
                    r"$\delta u_y$",
                    r"$\delta B_y$",
                    r"$\delta \pi$" if normalized else r"$\delta \Delta p$",
                ]
                self.equations = [wy_eq, bz_eq, wz_eq, by_eq, dp_eq]

        elif np.isclose(kx, 0.0):
            wx_lhs = "sigma*(dz(duz,2) -ky**2*duz)"
            wz_lhs = "sigma*dux"
            bx_lhs = "sigma*dbx"
            bz_lhs = "sigma*dbz"
            dp_lhs = "sigma*ddp"

            wx_rhs_adv = ""
            wz_rhs_adv = ""
            wx_rhs_pre = ""
            wz_rhs_pre = ""
            wx_rhs_lor = " -1j*ky*By*((ky**2*dbz -dz(dbz,2)) -(2*By**2 -1)/δ**2*dbz)"
            wz_rhs_lor = " +By*(1j*ky*dbx +By/δ*dbz)"
            wx_rhs_vis = ""
            wz_rhs_vis = ""
            wx_rhs_ani = ""
            wz_rhs_ani = ""
            if Pr > 0:
                wx_rhs_vis += " +ν*(dz(duz,4) -2*ky**2*dz(duz,2) +ky**4*duz)"
                wz_rhs_vis += " +ν*(dz(dux,2) -ky**2*dux)"
            if not self.isothermal:
                wx_rhs_ani += " +ky**2*By**2*(2*Bx/δ*ddp -dz(ddp))"
                wz_rhs_ani += " -1j*ky*Bx*By*ddp"
            if not np.isclose(self.Δβh, 0.0):
                wx_rhs_ani += " +Δβh*ky*By*(1j*(2*By**2 -1)*dz(dbz,2)" \
                            + " -6j*Bx*By**2/δ*dz(dbz)" \
                            + " +1j*(ky**2  -(2*By**2 -1)/δ**2)*dbz" \
                            + " +2*ky*Bx*By*dz(dbx)" \
                            + " +2*ky*By*(1 -3*Bx**2)/δ*dbx)"
                wz_rhs_ani += " -Δβh*By*(2*Bx*By*dz(dbz) +By/δ*dbz +1j*ky*(1 -2*Bx**2)*dbx)"

            bx_rhs_ind = " +By*(1j*ky*dux -By/δ*duz)"
            bz_rhs_ind = " +1j*ky*By*duz"
            bx_rhs_res = " +η*(dz(dbx,2) -ky**2*dbx)"
            bz_rhs_res = " +η*(dz(dbz,2) -ky**2*dbz)"
            bx_rhs_hal = ""
            bz_rhs_hal = ""
            if self.Hall:
                bx_rhs_hal += " +ϵ*By*( -dz(dbz,2) +(ky**2 -(2*By**2 - 1)/δ**2)*dbz)"
                bz_rhs_hal += " +ϵ*ky*By*(-ky*dbx +1j*By/δ*dbz)"

            if not self.isothermal:
                dp_rhs_adv = ""
                dp_rhs_str = " +By*Γβ*(By*dz(duz) -1j*ky*Bx*dux)"
                dp_rhs_res = " -2j*Γ2*η*By/δ*(Bx*dz(dbz,2)/ky -ky*Bx*dbz +By*dz(dbx))"
                dp_rhs_vis = ""

            wx_eq = f"{wx_lhs} ={wx_rhs_adv}{wx_rhs_pre}{wx_rhs_lor}{wx_rhs_vis}{wx_rhs_ani}"
            wz_eq = f"{wz_lhs} ={wz_rhs_adv}{wz_rhs_pre}{wz_rhs_lor}{wz_rhs_vis}{wz_rhs_ani}"
            bx_eq = f"{bx_lhs} ={bx_rhs_ind}{bx_rhs_res}{bx_rhs_hal}"
            bz_eq = f"{bz_lhs} ={bz_rhs_ind}{bz_rhs_res}{bz_rhs_hal}"
            if not self.isothermal:
                dp_eq = f"{dp_lhs} ={dp_rhs_adv}{dp_rhs_str}{dp_rhs_res}{dp_rhs_vis}"

            if self.isothermal:
                self.variables = ["dux", "duz", "dbx", "dbz"]
                self.labels = [
                    r"$\delta u_x$",
                    r"$\delta u_z$",
                    r"$\delta B_x$",
                    r"$\delta B_z$",
                ]
                self.equations = [wz_eq, wx_eq, bx_eq, bz_eq]
            else:
                self.variables = ["dux", "duz", "dbx", "dbz", "ddp"]
                self.labels = [
                    r"$\delta u_x$",
                    r"$\delta u_z$",
                    r"$\delta B_x$",
                    r"$\delta B_z$",
                    r"$\delta \Delta p$",
                ]
                self.equations = [wz_eq, wx_eq, bx_eq, bz_eq, dp_eq]

        else:
            raise NotImplementedError("Linearized Gyrotropic MHD equations for kx != 0 and ky != 0 are currently work in progress.")


        # Boundary conditions
        if self.periodic:
            self.boundaries = [ False ]*len(self.variables)
        else:
            self.boundaries = [ True  ]*len(self.variables)
            self.extra_binfo = [[f'dz({var}) -λ*{var}=0', f'dz({var}) +λ*{var}=0'] for var in self.variables]

        # Number of equations in system
        self.dim = len(self.variables)

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, a):
        self.__a = a
        self.δ   = a
        self.make_background()

    @property
    def S(self):
        return self.__S

    @S.setter
    def S(self, S):
        if S > 0.0:
            self.__S  = S
            self.η = 1/S
            self.ν = self.__Pr * self.η
        else:
            print('S must be > 0! Current value S={} is unchanged'.format(self.__S))

    @property
    def Pr(self):
        return self.__Pr

    @Pr.setter
    def Pr(self, Pr):
        if Pr >= 0.0:
            self.__Pr = Pr
            self.ν = self.__Pr * self.η
        else:
            print('Pr must be >= 0! Current value Pr={} is unchanged'.format(self.__Pr))

    @property
    def β(self):
        return self.__β

    @β.setter
    def β(self, β):
        if β >= 0.0:
            self.__β = β
            self.β0  = β
            self.Γβ  = self.Γ1 * self.β0 / 2 + self.Γ2 * self.Δβ0 / 2
        else:
            print(f"β must be >= 0! Current value β = {self.__β} is unchanged.")

    @property
    def Δβ(self):
        return self.__Δβ

    @Δβ.setter
    def Δβ(self, Δβ):
        self.__Δβ = Δβ
        self.Δβ0  = Δβ
        self.Δβh  = Δβ / 2
        self.Γβ  = self.Γ1 * self.β0 / 2 + self.Γ2 * self.Δβ0 / 2

    def make_background(self):
        from sympy import symbols, lambdify, diff, tanh, sech

        def sech_stable(x):
            """
            Stable hyperbolic secant for real x.
            Uses: sech(x) = 2*e^{-|x|} / (1 + e^{-2|x|})
            This avoids overflow for large |x| and loss of precision near 0.
            """
            import numpy as np

            x = np.asarray(x, dtype=np.float64)
            t = np.exp(-np.abs(x))          # in [0, 1]
            return (2 * t) / (1 + t*t)  # even function

        z  = symbols("z")
        zg = self.grid.zg

        a  = self.a
        z1 = self.z1
        z2 = self.z2

        if self.periodic:
            Bx_sym =  (tanh((z - z1) / a) + tanh((z - z1) / a)) / 2 \
                    - (tanh((z - z2) / a) + tanh((z - z2) / a)) / 2 - 1
            By_sym =  (sech((z - z1) / a) + sech((z - z1) / a)) / 2 \
                    - (sech((z - z2) / a) + sech((z - z2) / a)) / 2 - 1
        else:
            Bx_sym = tanh(z / a)
            By_sym = sech(z / a)

        modules = [{"sech": sech_stable}, "numpy"]

        self.Bx = lambdify(z, Bx_sym, modules)(zg)
        self.By = lambdify(z, By_sym, modules)(zg)
