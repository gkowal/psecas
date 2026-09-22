from psecas.string_methods import contains_symbol as _contains_symbol


class System:
    """
    Dedalus style initialization of an EVP problem.
    This will be useful for people for comparing with Dedalus.
    """

    def __init__(self, grid, variables, eigenvalue):

        self.grid = grid
        # Bind grid to the make_background method.
        # This ensures that the background is always evaluated with the
        # current grid resolution
        self.grid.bind_to(self.make_background)

        if type(variables) is str:
            self.variables = list([variables])
        else:
            self.variables = list(variables)

        self.equations = ['' for ii in range(len(self.variables))]
        self.boundaries = [False for ii in range(len(self.variables))]
        self.extra_binfo = [[None, None] for ii in range(len(self.variables))]

        self.substitutions = []

        self.labels = variables

        self.eigenvalue = eigenvalue

        # Create background
        self.make_background()

    @property
    def dim(self):
        return len(self.equations)

    def add_equation(self, eq, boundary=False):
        """
        Register a linearized equation.

        The equation is stored in the slot belonging to the variable that
        appears on its left-hand side, so exactly one variable must appear
        there.

        eq:       the equation, e.g. "sigma*f = dz(dz(f))"
        boundary: if True, impose Dirichlet conditions on this variable at
                  both ends. Use add_boundary() for anything else.
        """
        if '=' not in eq:
            raise ValueError(
                "The equation\n\n  {}\n\nhas no equal sign. Equations must "
                "be written as 'lhs = rhs', with the eigenvalue on the left."
                .format(eq)
            )

        lhs = eq.split('=')[0]

        found = None
        for ii, var in enumerate(self.variables):
            # Match whole identifiers only: a substring test would find the
            # variable 'vx' inside 'dvx' and claim the wrong slot.
            if _contains_symbol(lhs, var):
                if found is not None:
                    raise RuntimeError(
                        "Only one variable may appear on the LHS, but "
                        "'{}' and '{}' both appear in\n\n  {}"
                        .format(self.variables[found], var, lhs)
                    )
                found = ii

        if found is None:
            # Falling through silently used to leave this equation slot as
            # the empty string, and the failure then surfaced far away as
            # "IndexError: list index out of range" inside get_matrix1().
            raise ValueError(
                "None of the system variables {} appears on the left-hand "
                "side of\n\n  {}\n\nEvery equation must have exactly one "
                "variable on its LHS, since that is what decides which "
                "equation slot it fills. If the variable only enters through "
                "a substitution, write it out on the LHS."
                .format(self.variables, eq)
            )

        ii = found
        self.equations[ii] = eq
        if boundary:
            self.boundaries[ii] = True
            self.extra_binfo[ii] = ['Dirichlet', 'Dirichlet']
        else:
            self.boundaries[ii] = False
            self.extra_binfo[ii] = [None, None]

    def add_boundary(self, var, lower, upper):
        msg = 'Cannot set boundary on {}, as it is not found in system.variables'
        assert var in self.variables, msg.format(var)

        for ii, var2 in enumerate(self.variables):
            if var == var2:
                self.extra_binfo[ii] = [lower, upper]
                self.boundaries[ii] = True
                return

    def add_substitution(self, substitution):
        """Add equation substitution.
           There is not any symbolic manipulation,
           substitution will be done using simple text replacement.
        """
        assert '=' in substitution, 'should contain an equal sign'
        self.substitutions.append(substitution)

    def make_background(self):
        """
        Add problem parameters that depend on the grid using this fucntion.

        Example:
        Say the equations depend on rho(z) = exp(-(z/H0)**2).
        The following code would then be written:

        import numpy as np
        zg = self.grid.zg

        self.rho = np.exp(-(z/H0)**2)
        self.H0 = 0.4

        """
        pass
