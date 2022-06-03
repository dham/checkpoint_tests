from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
import numpy as np
from firedrake_adjoint import *
from pyadjoint.tape import no_annotations, Tape, set_working_tape
import time
import sys
PETSc.Sys.popErrorHandler()

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax = 1.22, 2.22

# Construct a circle mesh and then extrude into a cylinder:
mesh = Mesh('mesh/transfinite.msh') # This mesh was generated via gmshmesh
bottom_id, top_id = 1, 2
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)


# Define logging convenience functions:
def log(*args):
    """Log output to stdout from root processor only"""
    PETSc.Sys.Print(*args)


def log_params(f, str):
    """Log diagnostic paramters on root processor only"""
    if mesh.comm.rank == 0:
        f.write(str + "\n")
        f.flush()


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)
Told, Tnew = Function(Q, name="OldTemp"), Function(Q, name="NewTemp")

# Towards the bottom of the script are callbacks relating to the adjoint solutions (accessed through solve).
# We need to initiate the tape to make these work. 
tape = get_working_tape()


# Reference Initial State
# This will be used for "True Misfit" calculations
true_initial_state = Function(Q, name='TrueInitialState')

# Final states the like of tomography, note that we also load the reference profile 
final_state = Function(Q, name='RefTemperature')
final_state_file = DumbCheckpoint("../forward_100/Final_Temperature_State", mode=FILE_READ)
final_state_file.load(final_state, 'Temperature')
final_state_file.close()

# Initial condition, let's start with the final condition
Tic = Function(Q, name="T_IC")
Tic.project(final_state)

# Set up temperature field and initialise it with present day 
Told = Function(Q, name="OldTemperature")
Told.assign(Tic)
Tnew.assign(Told)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
Ttheta = 0.5*Tnew + (1 - 0.5)*Told

# Define time stepping parameters:
max_timesteps = 100

# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-7,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-6,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_rtol": 1e-6,
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
    }
}

# Energy Equation Solver Parameters:
energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-7,
    "pc_type": "sor",
}


# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(1e5)  # Rayleigh number
k = as_vector((X[0], X[1])) / r
C_ip = Constant(100.0)  # Fudge factor for interior penalty term used in weak imposition of BCs
p_ip = 2  # Maximum polynomial degree of the _gradient_ of velocity

# Temperature equation related constants:
delta_t = Constant(2.5e-5)  # Initial time-step
kappa = Constant(1.0)  # Thermal diffusivity

# Stokes equations in UFL form:
E = Constant(2.302585092994046)  # Activation energy for temperature dependent viscosity.
mu = exp( E * (0.5 - Tnew) )  # Viscosity
mu_f = Function(W, name="Viscosity")

stress = 2 * mu * sym(grad(u))
F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx + dot(n, v) * p * ds - (dot(v, k) * Ra * Ttheta) * dx
F_stokes += -w * div(u) * dx + w * dot(n, u) * ds  # Continuity equation

# nitsche free-slip BCs
F_stokes += -dot(v, n) * dot(dot(n, stress), n) * ds
F_stokes += -dot(u, n) * dot(dot(n, 2 * mu * sym(grad(v))), n) * ds
F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds

# Energy equation in UFL form:
F_energy = q * (Tnew - Told) / delta_t * dx + q * dot(u, grad(Ttheta)) * dx + dot(grad(q), kappa * grad(Ttheta)) * dx

# Temperature boundary conditions
bctb, bctt = DirichletBC(Q, 1.0, bottom_id), DirichletBC(Q, 0.0, top_id)

# Nullspaces and near-nullspaces:
x_rotV = Function(V).interpolate(as_vector((-X[1], X[0])))
V_nullspace = VectorSpaceBasis([x_rotV])
V_nullspace.orthonormalize()
p_nullspace = VectorSpaceBasis(constant=True)  # Constant nullspace for pressure n
Z_nullspace = MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])  # Setting mixed nullspace

# Generating near_nullspaces for GAMG:
nns_x = Function(V).interpolate(Constant([1., 0.]))
nns_y = Function(V).interpolate(Constant([0., 1.]))
V_near_nullspace = VectorSpaceBasis([nns_x, nns_y, x_rotV])
V_near_nullspace.orthonormalize()
Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1)])

# Write output files in VTK format:
u, p = z.split()  # Do this first to extract individual velocity and pressure fields.
u.rename("Velocity")
p.rename("Pressure")

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(F_stokes, z)  # velocity BC's handled through Nitsche
stokes_solver = NonlinearVariationalSolver(
    stokes_problem,
    solver_parameters=stokes_solver_parameters,
    appctx={"mu": mu},
#    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace
)
energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=energy_solver_parameters)

# Setting adjoint and forward callbacks, and control parameter:
control = Control(Tic)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Set Told = Tnew - assign the values of Tnew to Told
    Told.assign(Tnew)

## Initialise functional
functional = assemble(0.5*(Tnew - final_state)**2 * dx)

# Defining the object for pyadjoint
reduced_functional = ReducedFunctional(functional, control)

# Set up bounds, which will later be used to enforce boundary conditions in inversion:
T_lb = Function(Q, name="LB_Temperature")
T_ub = Function(Q, name="UB_Temperature")
T_lb.assign(0.0)
T_ub.assign(1.0)

# Taylor test:
Delta_temp = Function(Q, name="Delta_Temperature")
Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)*0.5
minconv = taylor_test(reduced_functional, Tic, Delta_temp)
print (minconv)
sys.exit(0)
