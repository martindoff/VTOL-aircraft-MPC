""" Data-driven Robust MPC of tiltwing VTOL aircraft.  

Forward transition (no wind)

This program computes a tube-based MPC control law for a VTOL aircraft according to the 
DC-TMPC algorithm in the paper: 'Data-driven Robust Model Predictive Control of Tiltwing 
Vertical Take-Off and Landing Aircraft' by Martin Doff-Sotta, Mark Cannon and Marko Bacic. 

The nonlinear dynamics of the aircraft is expressed as a difference of convex functions 
using polynomial algebraic techniques. The algorithm is based on successive linearisations
and exploits convexity to derive tight bounds on the linearisation errors and treat them 
as bounded disturbances in a robust tube-based MPC framework.  

(c) 03/2023 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""

import numpy as np
from scipy.linalg import block_diag, sqrtm, cholesky
import math
import cvxpy as cp
import mosek
import time
import os
import param_init as param
from terminal import get_terminal2
from vtol_model import feasibility2, interp_feas, linearise, f, seed_cost
from param_init import f1, f2
import DC_decomposition as DC
import pdb
import monomial
import numdifftools as nd
from control_custom import eul, dlqr, dp
import control as cs
try:
    import matplotlib
    import matplotlib.pyplot as plt 
    import matplotlib.patches as patches
    from matplotlib.ticker import FuncFormatter
except ImportError:
    pass

##########################################################################################
#################################### Initialisation ######################################
##########################################################################################

# Solver parameters
N = 110                                        # horizon 
T = 25                                         # terminal time
delta = T/N                                    # time step
tol1 = 10e-3                                   # tolerance   
maxIter = 1                                    # max number of iterations
d = 2                                          # polynomial half degree
N_eval = 50000                                 # number of evaluation points for LS fit
N_test = 500                                   # number of test points (DC and LS checks)
eps = np.finfo(float).eps                      # machine precision

# Variables initialisation
N_state = param.x_init.size                    # number of states
N_input = param.u_init.size                    # number of inputs
x = np.zeros((N_state, N+1))                   # state
pos = np.zeros((N_state, N+1))                 # position
x[:, 0] =  param.x_init
u = np.zeros((N_input, N))                     # control input
u_0=np.ones((N_input,N))*param.u_init[:, None] # (feasible) guess control input                     
x_0 = np.zeros((N_state, N+1))                 # (feasible) guess trajectory
x_r = np.ones_like(x)*param.h_r[:, None]       # reference trajectory 
t = np.zeros(N+1)                              # time vector 
K = np.zeros((N, N_input, N_state))            # gain matrix 
Phi1 = np.zeros((N, N_state, N_state))         # closed-loop state transition matrix of f1
Phi2 = np.zeros((N, N_state, N_state))         # closed-loop state transition matrix of f2
real_obj = np.zeros((N, maxIter+1))            # objective value
X_0 = np.zeros((N, maxIter+1, N_state, N+1))   # store state guess trajectories
U_0 = np.zeros((N, maxIter+1, N_input, N))     # store input guess trajectories
S_low = np.zeros((N, maxIter+1, N_state, N+1)) # store perturbed state (lower bound)
S_up = np.zeros((N, maxIter+1, N_state, N+1))  # store perturbed state (upper bound)
S = np.zeros((N, maxIter+1, N_state, N+1))     # store perturbed state    


##########################################################################################
################################### DC decomposition #####################################
##########################################################################################

# Predicate for AoA and speed limit for the generation of the samples
pred = lambda x: (x[:, 2] - np.arctan2(-x[:, 1], x[:, 0]) <=  param.alpha_max) &\
                 (x[:, 2] - np.arctan2(-x[:, 1], x[:, 0]) >=  param.alpha_min) &\
                 (param.V_e(param.V(x[:, 1], x[:, 0]), x[:, 3]) > 0.0001)

# Generate training samples
x_s = DC.f_rand(N_eval, param, pred) # generate random evaluation samples
F_s1, F_s2 = f1(*x_s.T), f2(*x_s.T)  # evaluate the functions at samples
avg = np.mean(x_s, axis=0)           # normalise data
std = np.std(x_s, axis=0)
#avg, std = np.zeros_like(avg), np.ones_like(std)

# Generate testing samples
x_test = DC.f_rand(N_test, param, pred)          # generate random evaluation samples
F_test1, F_test2 = f1(*x_test.T), f2(*x_test.T)  # evaluate the functions at samples

# DC split f1
print("Fitting on {} points".format(x_s.shape[0]))
print("###################  Computing DC decomposition of f1: #########################")
Q1, R1, P1, expo, D, D_ = DC.split(f1, d, x_s, F_s1, avg, std)  

# Test fit and split
DC.check(Q1, R1, P1, f1, x_test, expo, F_test1, avg, std)
#DC.plot(Q1, R1, P1, f1, expo, param, avg, std)  # visualise goodness of fit and split

# DC split f2
print("###################  Computing DC decomposition of f2: #########################")
Q2, R2, P2, expo, D, D_ = DC.split(f2, d, x_s, F_s2, avg, std) 

# Test fit and split
DC.check(Q2, R2, P2, f2, x_test, expo, F_test2, avg, std)
#DC.plot(Q2, R2, P2, f2, expo, param, avg, std)

# Sqrt
N_gram = R1.shape[0]
sqrt_R1 = sqrtm(R1)
sqrt_Q1 = sqrtm(Q1)
sqrt_R2 = sqrtm(R2)
sqrt_Q2 = sqrtm(Q2)
QR1, UR1 = np.linalg.qr(R1)
QR2, UR2 = np.linalg.qr(R2)

##########################################################################################
############################### Terminal set computation #################################
##########################################################################################

Q = param.Q
R = param.R
Q_N, gamma_N, K_hat = get_terminal2(param, delta, Q1, R1, Q2, R2, D_, expo, avg, std)
sqrt_Q_N = sqrtm(Q_N)
print("Terminal set parameters Q_hat, K_hat, gamma_hat :")
print("Q_N\n", Q_N)
print("K_hat\n", K_hat)
print("gamma_N\n", gamma_N)
##########################################################################################
################################# Feasible trajectory ####################################
##########################################################################################


    
d_feas = 0.1
x_feas, u_feas, t_feas = feasibility2(f, x[:, 0], x_r, d_feas, math.floor(T/d_feas), 
                                                param, Q1, R1, Q2, R2, D_, expo, avg, std)
                                                
t_0 = np.arange(N+1)*delta
x_0, u_0 = interp_feas(t_0, t_feas, x_feas, u_feas)

##########################################################################################
####################################### TMPC loop ########################################
##########################################################################################
avg_iter_time = 0
iter_count = 0

for i in range(N):

    print("Computation at time step {}/{}...".format(i+1, N)) 
    
    # Guess trajectory update
    if i > 0:
        #x_0[:, :-1] = x_0[:, 1:]
        x_0[:, :-1] = eul(f, u_0[:, :-1], x[:, i], delta, param) 
        A1_hat, B1_hat, A2_hat, B2_hat = linearise(x_0[:, -2, None], param.u_r[:, None], 
                                                delta, Q1, R1, Q2, R2, D_, expo, avg, std)  
        A_hat = A1_hat - A2_hat
        B_hat = B1_hat - B2_hat
        K_hat, _ = dp(A_hat[0, :, :], B_hat[0, :, :], Q, R, Q_N)
        u_0[:, -1, None] = K_hat @ ( x_0[:,-2, None]  - x_r[:, -2, None])\
                                                      + param.u_r[:, None]  # terminal u
        x_0[:, -1]  = x_0[:, -2] + delta*(f(x_0[:, -2], u_0[:, -1] , param))# terminal x 
    else:
        pass

    # Iteration
    k = 0 
    real_obj[i, 0] = 5000 
    delta_obj = 5000
    print('{0: <6}'.format('iter'), '{0: <5}'.format('status'), 
          '{0: <18}'.format('time'), '{}'.format('cost'))
    while k < maxIter:
    #while real_obj[i, k] > tol1 and k < maxIter and delta_obj > 0.1:
        
        # Linearise system at x_0, u_0
        A1, B1, A2, B2 = linearise(x_0[:, :-1], u_0, delta, Q1, 
                                   R1, Q2, R2, D_, expo, avg, std)  
        A = A1 - A2
        B = B1 - B2
        #print(x_0)
        
        # Compute K matrix (using dynamic programming)
        P = Q_N
        for l in reversed(range(N)): 
            K[l, :, :], P = dp(A[l, :, :], B[l, :, :], Q, R, P)
            Phi1[l, :, :] = A1[l, :, :] +  B1[l, :, :] @ K[l, :, :]
            Phi2[l, :, :] = A2[l, :, :] +  B2[l, :, :] @ K[l, :, :]
        
        # Quadratic approximation along trajectory
        z_0 = np.vstack([x_0[:, :-1], u_0])  # stack data
        start = time.time()
        g1, h1, g2, h2, Hess_Q1, Hess_R1, Hess_Q2, Hess_R2 = monomial.quad_approx(Q1, R1, 
                                                   Q2, R2, D_, z_0, expo, param, avg, std)
        end = time.time()
        #print("elapsed time:", end - start)
        
        """# check Hessian
        g1 = lambda x: monomial.polyval(expo, x, Q1, avg, std)
        h1 = lambda x: monomial.polyval(expo, x, R1, avg, std)
        g2 = lambda x: monomial.polyval(expo, x, Q2, avg, std)
        h2 = lambda x: monomial.polyval(expo, x, R2, avg, std)
        Hfun = nd.Hessian(h2)
        
        for l in range(N):
            print('Hessian true: ')
            print( Hess_R2[l, :, :])  # Hessian of y(x_)' R1 y(x_) wrt x_
            print('Hessian numerical: ')
            print(DC.hess(h2, ((np.hstack([x_0[:, l], u_0[:, l]]) )/1).T, .1))
            print('Hessian test numerical: ')
            print(Hfun(((np.hstack([x_0[:, l], u_0[:, l]]) )/1).T))"""
         
        # Check quadratic approx (at some randomly picked points)
        #monomial.plot(Q1, R1, Q2, R2, D_, x_test.T, expo, param, avg, std)
            
        # State transition of the closed loop
        Phi = Phi1 - Phi2
        
        ##################################################################################
        ############################ Optimisation problem ################################
        ##################################################################################
        
        N_ver = 2**N_state                     # number of vertices 
        
        # Optimisation variables
        y = cp.Variable((N_gram, N))
        theta = cp.Variable(N+1)               # cost 
        v = cp.Variable((N_input, N))          # input perturbation 
        s_low = cp.Variable((N_state, N+1))    # state perturbation (lower bound)
        s_up = cp.Variable((N_state, N+1))     # state perturbation (upper bound)
        s_ = {}                                # create dictionary for 3D variable 
        for l in range(N_ver):
            s_[l] = cp.Expression

        # Define blockdiag matrices for page-wise matrix multiplication
        K_ = block_diag(*K)
        Phi1_ = block_diag(*Phi1)
        Phi2_ = block_diag(*Phi2)
        B1_ = block_diag(*B1)
        B2_ = block_diag(*B2)
        
        # Wind disturbance (no wind)
        W_low = 0
        W_up = 0
        
        # Objective
        objective = cp.Minimize(cp.sum(theta))
        
        # Constraints
        constr = []
        
        # Assemble vertices
        s_[0] = s_low
        s_[1] = s_up
        s_[2] = cp.vstack([s_low[0, :], s_up[1, :]])
        s_[3] = cp.vstack([s_up[0, :], s_low[1, :]])
    
        for l in range(N_ver):
            # Define some useful variables
            s_r = cp.reshape(s_[l][:, :-1], (N_state*N,1))
            v_r = cp.reshape(v, (N_input*N,1))
            K_s = cp.reshape(K_ @ s_r, (N_input, N))
            Phi1_s = cp.reshape(Phi1_ @ s_r, ((N_state, N)))
            Phi2_s = cp.reshape(Phi2_ @ s_r, ((N_state, N)))
            B1_v = cp.reshape(B1_ @ v_r, (N_state, N))
            B2_v = cp.reshape(B2_ @ v_r, (N_state, N))
            
            # Objective constraints 
            constr += [theta[:-1] >= Q[0,0]*cp.abs(s_[l][0,:-1]+x_0[0,:-1]-x_r[0,:-1])\
                                   + Q[1,1]*cp.abs(s_[l][1,:-1]+x_0[1,:-1]-x_r[1,:-1])\
                       + R[0, 0]*cp.abs((v[0,:] + u_0[0,:] + K_s[0,:] - param.u_r[0]))\
                       + R[1, 1]*cp.abs((v[1,:] + u_0[1,:] + K_s[1,:] - param.u_r[1]))]                       
            constr += [theta[-1] 
                   >= (cp.norm(sqrt_Q_N @ (s_[l][:,-1] + x_0[:,-1] - x_r[:,-1])))]
            
            # Input constraints  
            constr += [v + u_0 + K_s >= param.u_min[:, None],
                       v + u_0 + K_s  <= param.u_max[:, None]]
            
            # Input rate constraints
            tol_cstr = 1e-6
            constr += [(cp.diff(v[0, :]+ u_0[0, :] + K_s[0, :], 2)/delta**2)*param.J_w\
                   <= param.M_max + tol_cstr]
            constr += [(cp.diff(v[0, :]+ u_0[0, :] + K_s[0, :], 2)/delta**2)*param.J_w\
                   >= param.M_min - tol_cstr]
            
            # Tube
            z = cp.vstack([x_0[:, :-1] + s_[l][:, :-1], u_0 + v + K_s])       
            constr += [s_low[:, 1:] <=  Phi1_s + B1_v\
                      - delta*cp.vstack([h1(z), h2(z)]) + delta*W_low]  #h(z) := h - h0
            
            constr += [s_up[:, 1:] >= s_[l][:, :-1] - Phi2_s - B2_v\
                       + delta*cp.vstack([g1(z), g2(z)]) + delta*W_up]  #g(z) := g - g0
                  
        # State constraints
        constr += [s_low[:, :-1] + x_0[:, :-1] >= param.x_min[:, None],
                  s_up[:, :-1] + x_0[:, :-1]  <= param.x_max[:, None], 
                  s_low[:, 0] == x[:, i] - x_0[:, 0], 
                  s_up[:, 0]  == x[:, i] - x_0[:, 0]] 
                        
        # Terminal set constraint 
        constr += [ np.sqrt(gamma_N) >= theta[-1]] 

        # Solve problem
        problem = cp.Problem(objective, constr)
        t_start = time.time()
        problem.solve(solver = cp.MOSEK, verbose=False)
       
        iter_time = time.time()-t_start
        avg_iter_time += iter_time
        print('{0: <5}'.format(k+1), '{0: <5}'.format(problem.status), 
              '{0: <5.2f}'.format(iter_time), '{0: <5}'.format(problem.value))
        if problem.status not in ["optimal"] and k > 0:
            print("Problem status {} at iteration k={}".format(problem.status, k))
            break
        
        ##################################################################################
        ############################### Iteration update #################################
        ##################################################################################
        # Save variables 
        S_low[i, k, :, :] = s_low.value.copy()
        S_up[i, k, :, :] = s_up.value.copy()
        X_0[i, k, :, :] = x_0.copy()
        U_0[i, k, :, :] = u_0.copy()
        x_0_old = x_0.copy()
        f_x = x_0.copy()
 
        # Input and state update
        s = np.zeros((N_state, N+1))
        s[:, 0] = x[:, i] - x_0[:, 0]  # implies s_0 = 0
        Ks = np.zeros_like(v.value)
        for l in range(N):
            Ks[:, l] =   K[l, :, :] @ s[:, l]
            u_0[:, l] += v.value[:, l] + Ks[:, l]
            f_x[:, l+1] = eul(f, u_0[:, l], x_0[:, l], delta, param)
            x_0[:, l+1] =  np.minimum(np.maximum(f_x[:, l+1], 
            x_0[:, l+1] + s_low[:, l+1].value), x_0[:, l+1] + s_up[:, l+1].value)
            s[:, l+1] = x_0[:, l+1]-x_0_old[:, l+1]
           
        S[i, k, :, :] = s.copy()  
        
        """plt.plot(t_0[:-1], (u_0[0, :])*180/np.pi)
        plt.ylabel('i_w')
        
        plt.figure()
        plt.plot(t_0[:-1], u_0[1, :])
        plt.ylabel('T')
        
        plt.figure()
        plt.plot(t_0, x_0[0, :], '-b')
        plt.plot(t_0, f_x[0, :], '-k')
        plt.plot(t_0, x_0_old[0, :], '--r')
        plt.plot(t_0, x_0_old[0, :] + s_low[0, ].value, '--g')
        plt.plot(t_0, x_0_old[0, :] + s_up[0, ].value, '--g')
        plt.ylabel('Vx')
        
        plt.figure()
        plt.plot(t_0, x_0[1, :], '-b')
        plt.plot(t_0, f_x[1, :], '-k')
        plt.plot(t_0, x_0_old[1, :], '--r')
        plt.plot(t_0, x_0_old[1, :] + s_low[1, ].value, '--g')
        plt.plot(t_0, x_0_old[1, :] + s_up[1, ].value, '--g')
        plt.ylabel('Vz')
        
        plt.show()"""
        
        # Step update 
        k += 1
        iter_count += 1
        real_obj[i, k] = problem.value
        delta_obj = real_obj[i, k-1]-real_obj[i, k]
        
    ######################################################################################
    #################################### System update ###################################
    ######################################################################################
    # Uncomment to exit at first iteration 
    """x = x_0
    u = u_0
    t = np.cumsum(np.ones(x.shape[1])*delta)-delta
    pos = np.hstack([np.zeros((2, 1)), np.cumsum(delta*x[:, :-1], axis=1)])
    x_r_0 = x_r
    break"""
    
    u[:, i] = u_0[:, 0]                                 # apply first input
    u_0[:, :-1] = u_0[:, 1:]                            # extract tail of the input
    x[:, i+1] = eul(f, u[:, i], x[:, i], delta, param)  # update nonlinear dynamics 
    t[i+1] = t[i] + delta
    pos[:, i+1] = pos[:, i] + delta*x[:, i]
    print('Speed:', x[:, i], 'Input (iw / T):', np.hstack([u[0,i]*180/np.pi, u[1,i]]))


##########################################################################################
##################################### Plot results #######################################
##########################################################################################
print('Average time per iteration: ', avg_iter_time/iter_count)
print('Average time per time step: ', avg_iter_time/i)

if not os.path.isdir('plot'):
    os.mkdir('plot')
    
#plt.rcParams['text.usetex'] = True  
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.unicode_minus'] = False
def math_formatter(x, pos):
    return "${}$".format(x).replace("-", u"\u2212")
      
# Trajectories 
fig, axs = plt.subplots(3, 1)
axs[0].plot(t, x[0,:], label=r'$V_x$')
axs[0].plot(t, X_0[0, 0, 0, :], '.' , label=r'$V_{x}^0$')
axs[0].plot(t, X_0[0, 0, 1, :], '.' , label=r'$V_{z}^0$')
axs[0].plot(t, x[1,:], label=r'$V_z$')
axs[0].plot(t, x_r[0,:], '-.', label=r'$V_x^r$')
axs[0].plot(t, x_r[1,:], '-.', label=r'$V_z^r$')
axs[0].legend(loc='upper right', prop={'size': 10})
axs[0].set(ylabel='Velocity (m/s)')
axs[1].plot(t[:-1], u[0,:]*180/np.pi, label=r'$i_w$')
axs[1].set(ylabel='Tiltwing angle (deg)')
axs[2].plot(t[:-1], u[1,:], label=r'$T$')
axs[2].set(xlabel='Time (s)', ylabel='Thrust T (N)')
fig.savefig('plot/tmpc1.eps', format='eps')
plt.savefig('plot/tmpc1.pdf', format='pdf')


plt.figure()
V = param.V(x[1, :], x[0, :])
gamma = param.gamma(x[1, :], x[0, :])
AoA = u[0, :] - gamma[:-1]
AoA_e = param.alpha_e(AoA, V[:-1], u[1, :]) 
plt.plot(t[:-1], AoA*180/np.pi, label=r'$\alpha$')
plt.plot(t[:-1], AoA_e*180/np.pi, label=r'$\alpha_e$')
plt.xlabel('Time (s)')
plt.ylabel('AoA (deg)')
plt.legend()

# Objective value
obj_init = seed_cost(X_0[0, 0, :, :], U_0[0, 0, :, :], Q, R, Q_N, param)
plt.figure()
plt.semilogy(range(0, N+1), np.hstack([obj_init, real_obj[:, 1]]))
plt.ylabel('Objective value $J$ at first iteration (-)')
plt.xlabel('Time step n (-)') 

# Objective value at first iteration
obj_init = seed_cost(X_0[0, 0, :, :], U_0[0, 0, :, :], Q, R, Q_N, param)
plt.figure()
plt.semilogy(range(0, maxIter+1), np.hstack([obj_init, real_obj[0, 1:]]))
plt.ylabel('Objective value $J$ at first time step (-)')
plt.xlabel('Iteration k (-)')

# Comparison solution with initial guess
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(t, x[0,:], label=r'$V_x$')
axs[0, 0].plot(t, X_0[0, 0, 0, :] ,label=r'$V_{x}^0$')
axs[0, 0].plot(t, x_r[0,:], '-.', label=r'$V_x^r$')
axs[0, 0].legend(loc='lower right', prop={'size': 10})
axs[0, 0].set_title('Forward velocity (m/s)')
axs[0, 0].set_xticks([])

axs[1, 0].plot(t, x[1,:], label=r'$V_z$')
axs[1, 0].plot(t, X_0[0, 0, 1, :], label=r'$V_{z}^0$')
axs[1, 0].plot(t, x_r[1,:], '-.', label=r'$V_z^r$')
axs[1, 0].legend(loc='lower right', prop={'size': 10})
axs[1, 0].set_title('Vertical velocity (m/s)')
axs[1, 0].set_xticks([])

axs[2, 0].plot(t[:-1], u[0,:]*180/np.pi, label=r'$i_w$')
axs[2, 0].plot(t[:-1], U_0[0, 0, 0, :]*180/np.pi, label=r'$\mu^0$')
#axs[2, 0].plot(t[:-1], gamma[:-1]*180/np.pi, label=r'$\gamma$')
axs[2, 0].legend(loc='upper right', prop={'size': 10})
axs[2, 0].set_title('Tiltwing angle (deg)')
axs[2, 0].set(xlabel='Time (s)')


axs[0, 1].plot(t[:-2], u[1,:-1], label=r'$T$')
axs[0, 1].plot(t[:-2], U_0[0, 0, 1, :-1], label=r'$\tau^0$')
axs[0, 1].legend(loc='upper right', prop={'size': 10})
axs[0, 1].set_title('Thrust (N)')
axs[0, 1].set_xticks([])

axs[1, 1].plot(t[:-1], AoA*180/np.pi, label=r'$\alpha$')
axs[1, 1].plot(t[:-1], AoA_e*180/np.pi, '-c', label=r'$\alpha_e$')
axs[1, 1].legend(loc='upper right', prop={'size': 10})
axs[1, 1].set_title('AoA (deg)')
axs[1, 1].set_xticks([])

pos_0 = np.hstack([np.zeros((2, 1)), np.cumsum(delta*X_0[0, 0, :, :-1], axis=1)])
axs[2, 1].plot(t, -pos[1, :], label=r'$h$')
axs[2, 1].plot(t, -pos_0[1, :], label=r'$h^0$')
axs[2, 1].set_title('Relative altitude (m)')
axs[2, 1].legend(loc='lower right', prop={'size': 10})
axs[2, 1].set(xlabel='Time (s)')

fig.tight_layout()
plt.savefig('plot/tmpc2.eps', format='eps')
plt.savefig('plot/tmpc2.pdf', format='pdf')

plt.show()