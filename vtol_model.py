""" VTOL aircraft model

(c) 03/2023 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import cvxpy as cp
import monomial
import scipy.linalg
from param_init import f1, f2
import control_custom as cs
import matplotlib.pyplot as plt 

def feasibility(f, x_0, x_r, delta, N, param, Q1, R1, Q2, R2, D, expo, avg, std):
    """ Genenerate a feasible trajectory for the VTOL """
    
    # Gains
    Ku = 50
    Tu = 29.7
    kp_x = 100
    ki_x = 2
    kp_z = 100
    ki_z = 1
    kd_z = 0
    kp_i = 0.6*Ku
    ki_i = 1.2*Ku/Tu
    kd_i = 0.075*Ku*Tu
    
    
    # Dimensions and initialisation
    N_state = x_0.shape[0]
    N_input = 2
    t = np.zeros((N+1, ))
    i_ref = np.zeros((N, ))
    diff_i = np.zeros((N, ))
    accel_i = np.zeros((N, ))
    Z = np.zeros((N+1, ))
    x = np.zeros((2, N+1))
    u = np.zeros((2, N))
    x[:, 0] = x_0
    V_x_r, V_z_r = x_r[0, 0], x_r[1, 0]
    sum_x, sum_z, sum_i = 0, 0, 0
    old_z = x_0[1]
    old_i_w = param.u_init[0]
    older_i_w = param.u_init[0]
    # Get data 
    arr=np.loadtxt('traj.txt')
    x2 = arr[0:2, : ]
    u2 = arr[2:4, : ]
    t2 = arr[4, : ]
    
    x_ref = np.zeros((2, N))
    u_ref = np.zeros((2, N))
    t_ref = np.arange(N)*delta
    for i in range(2):
        x_ref[i, :] = np.interp(t_ref, t2, x2[i, :])
    
    for i in range(2):
        u_ref[i, :] = np.interp(t_ref, t2, u2[i, :])
    
    # Linearise system around reference trajectory
    A1, B1, A2, B2 = linearise(x_ref, u_ref, delta, Q1, R1, Q2, R2, D, expo, avg, std)  
    A = A1 - A2
    B = B1 - B2
        
    # Compute K matrix (using dynamic programming)
    K = np.zeros((N, N_input, N_state))          # gain matrix 
    P = param.Q # change to Q_N later
    for l in reversed(range(N)): 
        K[l, :, :], P = cs.dp(A[l, :, :], B[l, :, :], param.Q, param.R, P)
               
    # Simulate the system
    for i in range(N):
        
        # Get states
        V_x, V_z = x[0, i], x[1, i]
        
        # Forward thrust 
        err_x = (V_x_r - V_x)
        sum_x += err_x*delta
        T_x = (kp_x*err_x + ki_x*sum_x)  # PI control
        
        # Vertical thrust
        h = Z[i]  # z coordinate
        V_z_r = 1*(0 - h)  # reference vertical speed
        err_z = -(V_z_r - V_z)
        sum_z += err_z*delta
        diff_z = -(V_z - old_z)/delta
        old_z = V_z
        T_z = (kp_z*err_z + ki_z*sum_z + kd_z*diff_z)/5  # PID control
        
        # Thrust
        T = np.maximum(np.minimum(np.sqrt(T_x**2 + T_z**2), param.T_max), param.T_min)
        
        # i_w
        i_ref[i] =  np.arctan2(T_z, T_x)  # reference tiltwing angle
        
        u[0, i] =i_ref[i]
        u[1, i] = T  # control input
        
        diff_i[i]  = (u[0, i] - old_i_w) / delta
        accel_i[i] = (u[0, i] - 2*old_i_w + older_i_w)/delta**2
        older_i_w = old_i_w
        old_i_w = u[0, i]
        
        # State update
        x[:, i+1] = cs.eul(f, u[:, i], x[:, i], delta, param)
        Z[i+1] = Z[i] + delta*V_z
        t[i+1] = t[i] + delta
    
        # Plot results
    plt.figure()
    plt.plot(t[:-1], u[0, :]*180/np.pi, '-b')
    plt.ylabel('i_w')
    plt.xlabel('Time (s)')
    
    plt.figure()
    plt.plot(t[:-1], u[1, :], '-b')
    plt.ylabel('Thrust')
    plt.xlabel('Time (s)')
    
    plt.figure()
    plt.plot(t, x[0, :], '-b')
    plt.ylabel('V_x')
    plt.xlabel('Time (s)')
        
    plt.figure()
    plt.plot(t, x[1, :], '-b')
    plt.ylabel('V_z')
    plt.xlabel('Time (s)')
        
    plt.figure()
    plt.plot(t, -Z, '-b')
    plt.ylabel('altitude variation')
    plt.xlabel('Time (s)')
    
    plt.figure()
    plt.plot(t[:-1], diff_i*180/np.pi, '-b')
    plt.ylabel('Tiltwing rate (deg/s)')
    plt.xlabel('Time (s)')
    
    plt.figure()
    plt.plot(t[:-1], accel_i*180/np.pi, '-b')
    plt.plot(t[:-1], np.ones_like(accel_i)*param.M_max/param.J_w, '--r')
    plt.ylabel('Tiltwing angular acceleration (deg/s^2)')
    plt.xlabel('Time (s)')
    
    plt.show()
    
    return x, u, t

def feasibility2(f, x_0, x_r, delta, N, param, Q1, R1, Q2, R2, D, expo, avg, std):
    """ Genenerate a feasible trajectory for the VTOL """
    
    
    # Dimensions and initialisation
    N_state = x_0.shape[0]
    N_input = 2
    t = np.zeros((N+1, ))
    i_ref = np.zeros((N, ))
    diff_i = np.zeros((N, ))
    accel_i = np.zeros((N, ))
    Z = np.zeros((N+1, ))
    x = np.zeros((2, N+1))
    u = np.zeros((2, N))
    x[:, 0] = x_0
    V_x_r, V_z_r = x_r[0, 0], x_r[1, 0]
    sum_x, sum_z, sum_i = 0, 0, 0
    old_z = x_0[1]
    old_i_w = param.u_init[0]
    older_i_w = param.u_init[0]
    
    # Get data 
    arr=np.loadtxt('traj.txt')
    x2 = arr[0:2, : ]
    u2 = arr[2:4, : ]
    t2 = arr[4, : ]
    
    x_ref = np.zeros((2, N))
    u_ref = np.zeros((2, N))
    t_ref = np.arange(N)*delta
    for i in range(2):
        x_ref[i, :] = np.interp(t_ref, t2, x2[i, :])
    
    for i in range(2):
        u_ref[i, :] = np.interp(t_ref, t2, u2[i, :])
    
    # Linearise system around reference trajectory
    A1, B1, A2, B2 = linearise(x_ref, u_ref, delta, Q1, R1, Q2, R2, D, expo, avg, std)  
    A = A1 - A2
    B = B1 - B2
        
    # Compute K matrix (using dynamic programming)
    Q = np.diag([1, 1])*0
    R = np.diag([1000, 0.001])
    K = np.zeros((N, N_input, N_state))          # gain matrix 
    P = Q # change to Q_N later
    for l in reversed(range(N)): 
        K[l, :, :], P = cs.dp(A[l, :, :], B[l, :, :], Q, R, P)
               
    # Simulate the system
    for i in range(N):
        
        # Get states
        V_x, V_z = x[0, i], x[1, i]
        
        u[:, i] = K[i, :, :] @ (x[:, i] -x_ref[:, i]) + u_ref[:, i]
        
        diff_i[i]  = (u[0, i] - old_i_w) / delta
        accel_i[i] = (u[0, i] - 2*old_i_w + older_i_w)/delta**2
        older_i_w = old_i_w
        old_i_w = u[0, i]
        
        # State update
        x[:, i+1] = cs.eul(f, u[:, i], x[:, i], delta, param)
        Z[i+1] = Z[i] + delta*V_z
        t[i+1] = t[i] + delta
        
    return x, u, t
    
def interp_feas(t_0, t_feas, x_feas, u_feas):
    
    N = t_0.shape[0]-1
    N_state = x_feas.shape[0]
    N_input = u_feas.shape[0]
    x_0 = np.zeros((N_state, N+1))
    u_0 = np.zeros((N_input, N))
    
    for i in range(N_state):
        x_0[i, :] = np.interp(t_0, t_feas, x_feas[i, :])
    
    for i in range(N_input):
        u_0[i, :] = np.interp(t_0[:-1], t_feas[:-1], u_feas[i, :])
        
    return x_0, u_0

def interp_K(t_0, t_feas, K):
    
    N = t_0.shape[0]-1
    N_state = K.shape[2]
    N_input = K.shape[1]
    K_0 = np.zeros((N+1, N_input, N_state))
    
    for i in range(N_input):
        for j in range(N_state):
            K_0[:, i, j] = np.interp(t_0, t_feas[:-1], K[:, i, j])
        
    return K_0
          
## Linearise model
def linearise(x_0, u_0, delta, Q1, R1, Q2, R2, D, expo, avg, std):
    """ Form the linearised discrete-time model around x_0, u_0 
    
    x[k+1] = (A1_d - A2_d) x[k] + (B1_d - B2_d) u[k]
    
    where A_d = A1_d - A2_d and B = B1_d - B2_d
    
    Input: 
    - x_0: guess state trajectory
    - u_0: guess input trajectory u_0
    - delta:  time step delta
    - Q1, R1, Q2, R2: Gram matrices of the DC decompsoition
    - D: derivative matrix of the vector of monomials s.t. d y / dx = D y
      
    Output: 
    - A1_d, B1_d, A2_d, B2_d: discrete-time matrices
    
    """
    # Dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    len_y = Q1.shape[0]  # length of monomial vector y
    
    # Initialisation
    Q = np.stack([Q1, Q2], axis=0)
    R = np.stack([R1, R2], axis=0)
    A1 = np.zeros((N, N_state, N_state))
    A2 = np.zeros((N, N_state, N_state))
    B1 = np.zeros((N, N_state, N_input))
    B2 = np.zeros((N, N_state, N_input))
    
    #             |Q1|                  |R1|                  |Q1|                  |R1|
    # A1 = d/dx y'|  | y    A2 = d/dx y'|  | y    B1 = d/du y'|  | y    B2 = d/du y'|  | y
    #             |Q2|                  |R2|                  |Q2|                  |R2|
    #
    x_ = np.hstack([x_0.T, u_0.T])
    for i in range(2):
        for j in range(2):
            A1[:, i, j] = monomial.polyval(expo, x_, D[j, :, :].T @ Q[i, :, :]\
                                                      + Q[i, :, :] @ D[j, :, :], avg, std)
            A2[:, i, j] = monomial.polyval(expo, x_, D[j, :, :].T @ R[i, :, :]\
                                                      + R[i, :, :] @ D[j, :, :], avg, std)
            
            B1[:, i, j] = monomial.polyval(expo, x_, D[2+j, :, :].T @ Q[i, :, :]\
                                                    + Q[i, :, :] @ D[2+j, :, :], avg, std)
            B2[:, i, j] = monomial.polyval(expo, x_, D[2+j, :, :].T @ R[i, :, :]\
                                                    + R[i, :, :] @ D[2+j, :, :], avg, std)
    
    # Linearised discrete-time model
    A1_d = np.eye(N_state) + delta*A1
    A2_d = delta*A2
    B1_d = delta*B1
    B2_d = delta*B2
    
    # Assemble Jacobians
    
    return A1_d, B1_d, A2_d, B2_d

# Dynamics
def f(x, u, param, w=np.zeros((2, 1))):
    """ Continuous-time system dynamics f such that
    
    dx/dt = f(x, u, w) s.t.
    
    d Vx /dt = f1(Vx, Vz, i_w, T) + w1
    d Vz /dt = f2(Vx, Vz, i_w, T) + w2
    
    Input: 
    - x: state
    - u: input
    - w: disturbance
    - param: structure of parameters
    
    Output: f dynamics function
    """
    
    x_ = np.hstack([x[0:2], u[0:2]])
    V_x_dot = f1(*x_) + w[0]
    V_z_dot = f2(*x_) + w[1]
    
    return np.array([V_x_dot[0], V_z_dot[0]], dtype=object)
 
def f_DC(x, u, param, w=np.zeros((2, 1))):
    """ Continuous-time system approximated dynamics g-h such that
    
    dx/dt = f(x, u, w) s.t.
    
    d Vx /dt = g1(Vx, Vz, i_w, T) - h1(Vx, Vz, i_w, T) + w1
    d Vz /dt = g2(Vx, Vz, i_w, T) - h2(Vx, Vz, i_w, T) + w2 
    
    Input: 
    - x: state
    - u: input
    - w: disturbance
    - param: structure of parameters
    
    Output: f dynamics function
    """
    
    x_ = np.hstack([x[0:2], u[0:2]])
    V_x_dot = monomial.polyval(param.expo, x_, param.Q1-param.R1, param.avg, param.std) + w[0]
    V_z_dot = monomial.polyval(param.expo, x_, param.Q2-param.R2, param.avg, param.std) + w[1]

    return np.array([V_x_dot[0], V_z_dot[0]], dtype=object)

def old_g(x, u, delta, param):


    """ Return the g convex dynamics from the DC decomposition
    
    f = g - h
    
    where f is the system dynamics and g, h are convex functions of the state / inputs
    
    Input: state x, input u, time step delta, parameter structure param
    Output: g convex dynamics function
    """
    
    N_gram = param.Q1.shape[0]
    N = u.shape[1]
    y = cp.vstack([np.ones((u[1, :]).shape[0]), u[1, :], u[0, :], x[1, :], x[0, :]])
    
    z_1 = cp.multiply(cp.vec(y), cp.vec(param.Q1 @ y))
    z_2 = cp.multiply(cp.vec(y), cp.vec(param.Q2 @ y))
    w_1 = cp.reshape(z_1, (N_gram, N))
    w_2 = cp.reshape(z_2, (N_gram, N))
    
    g_1 = cp.sum(w_1, axis=0)
    g_2 = cp.sum(w_2, axis=0)
    
    return cp.vstack([g_1, g_2]) 


def old_h(x, u, delta, param):


    """ Return the h convex dynamics from the DC decomposition
    
    f = g - h
    
    where f is the system dynamics and g, h are convex functions of the state / inputs
    
    Input: state x, input u, time step delta, parameter structure param
    Output: h convex dynamics function
    """
    
    N_gram = param.R1.shape[0]
    N = u.shape[1]
    y = cp.vstack([np.ones((u[1, :]).shape[0]), u[1, :], u[0, :], x[1, :], x[0, :]])
    
    z_1 = cp.multiply(cp.vec(y), cp.vec(param.R1 @ y))
    z_2 = cp.multiply(cp.vec(y), cp.vec(param.R2 @ y))
    w_1 = cp.reshape(z_1, (N_gram, N))
    w_2 = cp.reshape(z_2, (N_gram, N))
    
    h_1 = cp.sum(w_1, axis=0)
    h_2 = cp.sum(w_2, axis=0)
    
    return cp.vstack([h_1, h_2])

def h2(x, u, param):
    """ Return the h convex dynamics from the DC decomposition
    
    f = g - h
    
    where f is the system dynamics and g, h are convex functions of the state / inputs
    
    Input: state x, input u, time step delta, parameter structure param
    Output: h convex dynamics function
    """
    
    if u.ndim < 2:
        y = np.vstack([1, u[1], u[0], x[1], x[0]])
        return np.vstack([y.T @  param.R1 @ y, y.T @  param.R2 @ y])[:, 0]
         
    N_gram = param.R1.shape[0]
    N = u.shape[1]
    h = np.zeros((2, N))
    y = np.vstack([np.ones((u[1, :]).shape[0]), u[1, :], u[0, :], x[1, :], x[0, :]])
    for i in range(N):
        h[0, i] = y[:, i] @  param.R1 @ y[:, i]
        h[1, i] = y[:, i] @  param.R2 @ y[:, i]
    
    return h #cp.vstack([h_1, h_2])

def g2(x, u, param):
    """ Return the g convex dynamics from the DC decomposition
    
    f = g - h
    
    where f is the system dynamics and g, h are convex functions of the state / inputs
    
    Input: state x, input u, time step delta, parameter structure param
    Output: h convex dynamics function
    """
    
    if u.ndim < 2:
        y = np.vstack([1, u[1], u[0], x[1], x[0]])
        #print(np.vstack([y.T @  param.Q1 @ y, y.T @  param.Q2 @ y])[:, 0])
        return np.vstack([y.T @  param.Q1 @ y, y.T @  param.Q2 @ y])[:, 0]
        
    N_gram = param.Q1.shape[0]
    N = u.shape[1]
    g = np.zeros((2, N))
    y = np.vstack([np.ones((u[1, :]).shape[0]), u[1, :], u[0, :], x[1, :], x[0, :]])
    for i in range(N):
        g[0, i] = y[:, i] @  param.Q1 @ y[:, i]
        g[1, i] = y[:, i] @ param.Q2 @ y[:, i]
    
    return g #cp.vstack([h_1, h_2])
    
def g_minus_h(x, u, param):
    print(g2(x, u, param), h2(x, u, param), g2(x, u, param)-h2(x, u, param))
    return g2(x, u, param) - h2(x, u, param)
    
def seed_cost(x_0, u_0, Q, R, Q_N, param):
    """ Compute cost 
    Input: trajectories x0 and u0, penalty matrices Q, R, Q_N, parameter structure param
    Output: cost J 
    """
    
    J = 0
    N = u_0.shape[1]
    for k in range(N):
        J +=  np.diag(Q) @ np.abs(x_0[:, k]-param.h_r)\
              + np.diag(R) @  np.abs(u_0[:, k]- param.u_r)
    
    # Terminal cost term
    J += np.sqrt((x_0[:, -1]-param.h_r).T @ Q_N @ (x_0[:, -1]-param.h_r))
    
    return J