"""
    A minimal test for checkpointing to disk The problem is advection-diffusion
    for stokes flow After a given number of time steps, the functional is
    constructed by an L2 difference of temperature field and a reference field.

    In the end we compute the functional and gradient for two arbitrary
    controls.
"""
from firedrake import *
from firedrake.petsc import PETSc
from firedrake_adjoint import Control, ReducedFunctional, stop_annotating, \
    get_working_tape
import os
from firedrake.adjoint.checkpointing import  \
    enable_disk_checkpointing
from progress.bar import FillingSquaresBar



# Reference temperature field file:
ref_fi_name = "./reference_temperature.h5"

# Checking if the file exists
if not os.path.isfile(ref_fi_name):
    raise ValueError(f"Assembly of the functional needs: {ref_fi_name}")

# logging.set_log_level(1)
# logging.set_level(1)

# Geometric Constants:
x_max, y_max, disc_n = 1.0, 1.0, 100

enable_disk_checkpointing(cleanup=False)
# and Interval mesh of unit size
# mesh1d = IntervalMesh(disc_n, length_or_left=0.0, right=x_max)
# # extruding the base mesh "mesh1d" in the third dimension
# mesh = ExtrudedMesh(mesh=mesh1d, layers=disc_n, layer_height=y_max/disc_n,
#                     extrusion_type='uniform', kernel=None, gdim=None)

with CheckpointFile(ref_fi_name, "r") as infile:
    mesh = infile.load_mesh("firedrake_default_extruded")
    final_state = infile.load_function(mesh, "RefTemperature")


# Print function to ensure log output is only written on processor zero (if
# running in parallel) ####
def log(*args):
    if mesh.comm.rank == 0:
        PETSc.Sys.Print(*args)


# Top and bottom ids, for extruded mesh
top_id, bottom_id = 'top', 'bottom'
left_id, right_id = 1, 2

# spatial coordinates
X = x, y = SpatialCoordinate(mesh)

# setting up vertical direction
y_abs = sqrt(y**2)
yhat = as_vector((0, y)) / y_abs

# Global Constants:
Ra = Constant(1e7)
max_num_timesteps = 4
delta_t = Constant(5e-6)
kappa = Constant(1.0)

# Below are callbacks relating to the adjoint solutions (accessed through
# solve). Not sure what the best place would be to initiate working tape!
tape = get_working_tape()
tape.progress_bar = FillingSquaresBar

# Temporal discretisation - Using a Crank-Nicholson scheme where theta_ts =
# 0.5:
theta_ts = 0.5

# Geometry and Spatial Discretization:
# Set up function spaces - currently using the P2P1 element pair :
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)

# Set up mixed function space and associated test functions:
Z = MixedFunctionSpace([V, W])
N, M = TestFunctions(Z)
Y = TestFunction(Q)

# Set up fields on these function spaces - split into each component so that
# they are easily accessible:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)     # can we nicely name mixed function space fields?

# Final states the like of tomography, note that we also load the reference
# # profile.
# final_state = Function(Q, name='RefTemperature')

# final_state_file = DumbCheckpoint("../../forward/State_100", mode=FILE_READ)
# final_state_file.load(final_state, 'Temperature')
# final_state_file.close()

# Initial condition, let's start with the final condition
Tic = Function(Q, name="T_IC")
Tic.project(final_state)

# Set up temperature field and initialise it with present day.
Told = Function(Q, name="OldTemperature")
Told.assign(Tic)

Tnew = Function(Q, name="Temperature")
Tnew.assign(Told)

# Temporal discretisation - Using a theta scheme:
T_theta = theta_ts * Tnew + (1-theta_ts) * Told

# ********************** Setup Equations ************************
# viscosity
mu = Constant(1.0)   # Constant viscosity


# deviatoric stresses
def tau(u):
    return mu * (grad(u)+transpose(grad(u)))


# Stokes in weak form.
F_stokes = inner(grad(N), tau(u))*dx - div(N)*p*dx
F_stokes += - (dot(N, yhat)*Ra*T_theta)*dx
F_stokes += - div(u) * M * dx

# Setting No-slip BC for top and bottom
bcu_topbase = DirichletBC(Z.sub(0), 0.0, (top_id, bottom_id))
bcu_rightleft = DirichletBC(Z.sub(0), 0.0, (left_id, right_id))

# Pressure nullspace
p_nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)]
)

# Temperature, advection-diffusion equation
F_energy = Y * ((Tnew - Told) / delta_t) * dx \
                + Y*dot(u, grad(T_theta)) * dx \
                + dot(grad(Y), kappa*grad(T_theta)) * dx

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(
    F_stokes, z, bcs=[bcu_topbase, bcu_rightleft]
)
stokes_solver = NonlinearVariationalSolver(
    stokes_problem, nullspace=p_nullspace, transpose_nullspace=p_nullspace
)

energy_problem = NonlinearVariationalProblem(F_energy, Tnew)
energy_solver = NonlinearVariationalSolver(energy_problem)

# Setting adjoint and forward callbacks, and control parameter
control = Control(Tic)

# Now perform the time loop:
for timestep in FillingSquaresBar("Forward solve").iter(range(max_num_timesteps)):
    # Solve system - configured for solving non-linear systems, where
    # everything is on the LHS (as above) and the RHS == 0.
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Set Told = Tnew - assign the values of Tnew to Told
    Told.assign(Tnew)

# Initialise functional
functional = assemble(0.5*(Tnew - final_state)**2 * dx)

# Output file
grd_file = File("gradients.pvd")

# Constructing the ReducedFunctional
reduced_functional = ReducedFunctional(functional, control)
print(tape._checkpoint_metadata.current_checkpoint_file)
print(tape._checkpoint_metadata.init_checkpoint_file)
import networkx as nx
nx.nx_agraph.to_agraph(tape.create_graph()).draw("tape.pdf", prog="dot")
# Calculations of RF and gradient for two arbitrary control values for Tic
# The gradients are writted out for visualisation
with stop_annotating():
    mystr = "Functional Values: "
    Q_gradient = Function(Q, name="gradient")
    for control_value in [
        0.5 - 0.1*exp(-0.5*((X-as_vector((0.5, 0.40)))/Constant(0.03))**2),
        Constant(0.2)
    ]:
        Tic.interpolate(control_value)
        mystr += f" * {reduced_functional(Tic)} *"
        Q_gradient.interpolate(reduced_functional.derivative())
        grd_file.write(Q_gradient)
    log(mystr)

