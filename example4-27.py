# Based on mhe_spring_damper exampel by Kurt Geebelen

import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
import casadi.tools as ctools
from scipy import linalg
#plt.interactive(True)
np.random.seed(0)

# Settings of the filter
N = 10 # Horizon length
dt = 0.25 # Time step
Nint = 2 # Number of integrator steps per control interval

sigma_p = 0.25 # Standard deviation of the measurements
sigma_w = 0.001 # Standard deviation for the process noise
R = casadi.DMatrix(1/sigma_p**2) # resulting weighting matrix for the position measurements
Q = casadi.DMatrix(1/sigma_w**2) # resulting weighting matrix for the process noise

Nsimulation = 120 # Lenght of the simulation

X_0 = casadi.vertcat([1.0,0.0,4.0])
X0 = casadi.vertcat([0.5,0.05,0.0])

# Parameters of the system
k1= 0.5
k_1 = 0.05
k2=0.2
k_2=0.01

RT = 32.84

C = casadi.horzcat([1.0,1.0,1.0])*RT

# The state
states = ctools.struct_symSX(["cA","cB","cC"]) # Full state vector of the system: position x and velocity dx
Nstates = states.size # Number of states
# Set up some aliases
cA, cB, cC = states[...]

# The control input
controls = ctools.struct_symSX(["F"]) # Full control vector of the system
Ncontrols = controls.size # Number of control inputs
# Set up some aliases
F, = controls[...]

# Disturbances
disturbances = ctools.struct_symSX(["w"]) # Process noise vector
Ndisturbances = disturbances.size # Number of disturbances
# Set up some aliases
w, = disturbances[...]

# Measurements
measurements = ctools.struct_symSX(["y"]) # Measurement vector
Nmeas = measurements.size # Number of measurements
# Set up some aliases
y, = measurements[...]

# Create Structure for the entire horizon

# Structure that will be degrees of freedom for the optimizer
shooting = ctools.struct_symSX([(
    ctools.entry("X",repeat=N,struct=states),
    ctools.entry("W",repeat=N-1,struct=disturbances)
)])

# Structure that will be fixed parameters for the optimizer
parameters = ctools.struct_symSX([(
    ctools.entry("U",repeat=N-1,struct=controls),
    ctools.entry("Y",repeat=N,struct=measurements),
    ctools.entry("S",shape=(Nstates,Nstates)),
    ctools.entry("x0",shape=(Nstates,1))
)])
S = parameters["S"]
x0 = parameters["x0"]
# Define the ODE right hand side
A = casadi.blockcat([[-1.0,0.0],[1.0,-2.0],[1.0,1.0]])
B = casadi.vertcat([k1*cA-k_1*cB*cC,k2*cB**2-k_2*cC])

rhs = casadi.mul(A,B)

f = casadi.SXFunction([states,controls,disturbances],[rhs])
f.init()

# Build an integrator for this system: Runge Kutta 4 integrator
k1 = f.call([states,controls,disturbances])[0]
k2 = f.call([states+dt/Nint/2.0*k1,controls,disturbances])[0]
k3 = f.call([states+dt/Nint/2.0*k2,controls,disturbances])[0]
k4 = f.call([states+dt/Nint*k3,controls,disturbances])[0]

states_1 = states+dt/Nint/6.0*(k1+2*k2+2*k3+k4)
phi = casadi.SXFunction([states,controls,disturbances],[states_1])
phi.init()

# Repeat this block Nint times
states_N = states
for i in range(Nint):
  states_N = phi.call([states_N,controls,disturbances])[0]
  
phi = casadi.SXFunction([states,controls,disturbances],[states_N])
phi.init()
  
PHI = phi.jacobian()
PHI.init()
# Define the measurement system
h = casadi.SXFunction([states],[casadi.mul(C,states)]) # We have measurements of concentrations
h.init()
H = h.jacobian()
H.init()
# Build the objective
obj = 0
# First the arrival cost
obj += casadi.mul([(shooting["X",0]-parameters["x0"]).T,S,(shooting["X",0]-parameters["x0"])])
#Next the cost for the measurement noise
for i in range(N):
  vm = h.call([shooting["X",i]])[0]-parameters["Y",i]
  obj += casadi.mul([vm.T,R,vm])
#And also the cost for the process noise
for i in range(N-1):
  obj += casadi.mul([shooting["W",i].T,Q,shooting["W",i]])

# Build the multiple shooting constraints
g = []
for i in range(N-1):
  g.append( shooting["X",i+1] - phi.call([shooting["X",i],parameters["U",i],shooting["W",i]])[0] )

# Formulate the NLP
nlp = casadi.SXFunction(casadi.nlpIn(x=shooting,p=parameters),casadi.nlpOut(f=obj,g=casadi.vertcat(g)))

# Make a simulation to create the data for the problem
simulated_X = np.matrix(np.zeros((Nstates,Nsimulation)))
simulated_X[:,0] = X0 # Initial state
t = np.linspace(0,(Nsimulation-1)*dt,Nsimulation) # Time grid
simulated_U = np.matrix(np.cos(t)) # control input for the simulation
simulated_U[:,Nsimulation/2:] = 0.0
simulated_W = sigma_w*np.random.randn(Ndisturbances,Nsimulation-1) # Process noise for the simulation
for i in range(Nsimulation-1):
  phi.setInput(simulated_X[:,i],0)
  phi.setInput(simulated_U[:,i],1)
  phi.setInput(simulated_W[:,i],2)
  phi.evaluate()
  simulated_X[:,i+1] = phi.getOutput(0)
#Create the measurements from these states
simulated_Y = np.zeros((Nmeas,Nsimulation)) # Holder for the measurements
for i in range(Nsimulation):
  h.setInput(simulated_X[:,i],0)
  h.evaluate()
  simulated_Y[:,i] = h.getOutput(0)
# Add noise the the position measurements
simulated_Y += sigma_p*np.random.randn(simulated_Y.shape[0],simulated_Y.shape[1])

#The initial estimate and related covariance, which will be used for the arrival cost
sigma_x0 = 0.5
P = sigma_x0**2*np.matrix(np.eye(Nstates))
x0 = X0
# Create the solver
#nlp_solver = IpoptSolver(nlp)
nlp_solver = casadi.NlpSolver("ipopt",nlp)
nlp_solver.setOption({"print_level":0, "print_time": False})
#nlp_solver.setOption('linear_solver','MA57')
nlp_solver.setOption('max_iter',100)
nlp_solver.setOption('expand',True)
nlp_solver.init()

# Set the bounds for the constraints: we only have the multiple shooting constraints, so all constraints have upper and lower bound of zero
nlp_solver.setInput(0,"lbg")
nlp_solver.setInput(0,"ubg")

# Create a holder for the estimated states and disturbances
estimated_X = casadi.DMatrix.zeros(Nstates,Nsimulation)
estimated_W = casadi.DMatrix.zeros(Ndisturbances,Nsimulation-1)
# For the first instance we run the filter, we need to initialize it.
current_parameters = parameters(0)
current_parameters["U",casadi.horzcat] = simulated_U[:,0:N-1]
current_parameters["Y",casadi.horzcat] = simulated_Y[:,0:N]
current_parameters["S"] = linalg.inv(P) # Arrival cost is the inverse of the initial covariance
current_parameters["x0"] = x0
initialisation_state = shooting(0)
initialisation_state["X",casadi.horzcat] = simulated_X[:,0:N]

nlp_solver.setInput(current_parameters,"p")
nlp_solver.setInput(initialisation_state,"x0")

nlp_solver.evaluate()
# Get the solution
solution = shooting(nlp_solver.output("x"))
estimated_X[:,0:N] = solution["X",casadi.horzcat]
estimated_W[:,0:N-1] = solution["W",casadi.horzcat]

t0 = time.time()
# Now make a loop for the rest of the simulation
for i in range(1,Nsimulation-N+1):
  # Update the arrival cost, using linearisations around the estimate of MHE at the beginning of the horizon (according to the 'Smoothed EKF Update'): first update the state and covariance with the measurement that will be deleted, and next propagate the state and covariance because of the shifting of the horizon
  print "step %d/%d (%s)" % (i, Nsimulation-N , nlp_solver.getStat("return_status"))
  H.setInput(solution["X",0],0)
  H.evaluate()
  H0 = H.getOutput(0)
  K = casadi.mul([P,H0.T,linalg.inv(casadi.mul([H0,P,H0.T])+R)])
  P = casadi.mul((casadi.DMatrix.eye(Nstates)-casadi.mul(K,H0)),P)
  h.setInput(solution["X",0],0)
  h.evaluate()
  x0 = x0 + casadi.mul(K, current_parameters["Y",0]-h.getOutput(0)-casadi.mul(H0,x0-solution["X",0]) )
  phi.setInput(x0,0)
  phi.setInput(current_parameters["U",0],1)
  phi.setInput(solution["W",0],2)
  phi.evaluate()
  x0 = phi.getOutput(0)
  PHI.setInput(solution["X",0],0)
  PHI.setInput(current_parameters["U",0],1)
  PHI.setInput(solution["W",0],2)
  PHI.evaluate()
  F = PHI.getOutput(0)
  PHI.evaluate()
  P = casadi.mul([F,P,F.T]) + linalg.inv(Q)
  # Get the measurements and control inputs 
  current_parameters["U",casadi.horzcat] = simulated_U[:,i:i+N-1]
  current_parameters["Y",casadi.horzcat] = simulated_Y[:,i:i+N]
  current_parameters["S"] = linalg.inv(P)
  current_parameters["x0"] = x0
  # Initialize the system with the shifted solution
  initialisation_state["W",casadi.horzcat,0:N-2] = estimated_W[:,i:i+N-2] # The shifted solution for the disturbances
  initialisation_state["W",N-2] = casadi.DMatrix.zeros(Ndisturbances,1) # The last node for the disturbances is initialized with zeros
  initialisation_state["X",casadi.horzcat,0:N-1] = estimated_X[:,i:i+N-1] # The shifted solution for the state estimates
  # The last node for the state is initialized with a forward simulation
  phi.setInput(initialisation_state["X",N-1] ,0)
  phi.setInput(current_parameters["U",-1],1)
  phi.setInput(initialisation_state["W",-1],2)
  phi.evaluate()
  initialisation_state["X",N-1] = phi.getOutput(0)
  # And now initialize the solver and solve the problem
  nlp_solver.setInput(current_parameters,"p")
  nlp_solver.setInput(initialisation_state,"x0")
  nlp_solver.evaluate()
  # Now get the state estimate. Note that we are only interested in the last node of the horizon
  estimated_X[:,N-1+i] = solution["X",N-1]
  estimated_W[:,N-2+i] = solution["W",N-2]
# Plot the results

print "elapsed time = %.3f [s]" % (time.time()-t0)

for i,l in enumerate(states.keys()):
  plt.figure(1)
  plt.plot(t,casadi.vec(estimated_X[i,:]),'b--')
  plt.plot(t,casadi.vec(simulated_X[i,:]),'r--')
  plt.title(l)
  plt.xlabel('Time')
  plt.legend(['Estimated','Real'])
  plt.grid()


  plt.figure(2)
  plt.plot(t,casadi.vec(estimated_X[i,:]-simulated_X[i,:]),'b--')
  plt.title("error on " + l)
  plt.xlabel('Time')
  plt.legend(['Error between estimated and real'])
  plt.grid()

plt.show()

error = estimated_X[0,:]-simulated_X[0,:]
print casadi.mul(error,error.T)
assert(casadi.mul(error,error.T)<0.01)