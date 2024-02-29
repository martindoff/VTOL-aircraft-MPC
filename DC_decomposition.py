""" DC-decomposition

Achieve decomposition of the nonlinear dynamics as a difference of convex functions

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
from math import comb
import scipy.linalg
import least_squares as ls
import cvxpy as cp
import monomial
import param_init as p
import sys
import time
import copy

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker  
except ImportError:
    pass
np.set_printoptions(threshold=sys.maxsize)

## Cartesian product
def cartesian_product(*arrays):
    """
    Generate the n-ary cartesian product of the n input arrays
    
    Input: unpacked input arrays
    Output: matrix whose rows are ordered n-tuples formed from the input 
    """
    
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
    
## Function evaluation points
def f_arg(N_s, p, filtered_index):
    """
    Generate evaluation points for the problem.
    
    Input: number of samples per state variable N_s, parameter structure param
    Output: evaluation points
    """
    # Generate an array for each state variable
    ctr = p.ctr
    n = ctr.shape[0]
    X = np.zeros((n, N_s))
    for i in range(n):
        X[i,:] = np.linspace(ctr[i, 0], ctr[i, 1], N_s)
    
    # Compute cartesian product
    x_s = cartesian_product(*X)
    
    # Check for condition on AoA
    return x_s[filtered_index(x_s), :]

## Random function evaluation points
def f_rand(N_s, p, filtered_index):
    """
    Generate random evaluation points for the problem.
    
    Input: number of samples N_s, parameter structure param
    Output: evaluation points
    """
    
    ctr = p.ctr
    n = ctr.shape[0]
    x_s = np.zeros((N_s, n))
    for i in range(n):
        x_s[:, i] = np.random.rand(N_s)*(ctr[i, 1]-ctr[i, 0]) + ctr[i, 0]

    # Check for condition on AoA  
    return x_s[filtered_index(x_s), :]
           
## Optimisation
def regularize(P, Q, R, solver_name="MOSEK"):
    """" Regularize the solution obtained for Q and R. 
    
    Input: Gram matrix P of polynomial f, Gram matrix Q of polynomial g, Gram matrix R of
    polynomial h, solver name (default is solver_name="MOSEK")
    Ouput: (Regularized) Gram matrices Q, R of polynomials g, h s.t. f = g - h and g, h are 
    convex
    """
    
    # Define problem dimension
    N = P.shape[0]
    
    # Define optimisation variables 
    Q0 = cp.Variable((N, N), symmetric=True)
    dQ = cp.Variable((N, N), symmetric=True) 
    
    # Define objective
    obj = cp.Minimize(cp.norm((Q0)))
    
    # Constraints
    constr = []
    constr += [Q == Q0 + dQ]
    
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=True, solver = solver_name_map[solver_name])
    
    print("Regularization status: ", prob.status)
         
    return Q0.value, Q0.value-P
    
def cvx(D, y_s, F_s, solver_name="MOSEK"):
    """" Compute the DC decomposition of a polynomial f from samples F_s.
    
    Let F_s = f(x_s) be a sequence of samples of f evaluated at x_s and y_s = y(x_s) be
    the vector of monomials of degree up to d in the Gram form. 
     
    The function performs the DC decomposition such that 
    f = y_s'(Q-R)y where g = y'Qy, h = y'Ry are SOS convex functions. 
    
    Input: matrix of derivatives D, vector of monomials of degree up to d in the Gram form
    evaluated at sample points y_s, samples of polynomial values F_s, solver name (default
    is solver_name="MOSEK")
    
    Ouput: Gram matrices Q, R of polynomials g, h s.t. f = g - h and g = y'Qy, h = y'Ry 
    are convex. 
    """
    
    # Define problem dimension
    N = y_s.shape[1]
    N_s = y_s.shape[0]
    n = D.shape[0]
    
    # Stack data
    y = y_s.T
    Y = scipy.linalg.block_diag(*y_s)
    
    # Define optimisation variables 
    Q = cp.Variable((N, N), symmetric=True)
    R = cp.Variable((N, N), symmetric=True)
    m = cp.Variable(N_s)
    t = cp.Variable(1)
    
    # Construct Hessians
    H_Q, H_R = [], []
    for i in range(n):
        row_Q, row_R = [], []
        for j in range(n):
            blck_Q = D[i, :, :].T @ Q @ D[j, :, :] + D[j, :, :].T @ Q @ D[i, :, :]\
                   + Q @ D[j, :, :] @ D[i, :, :] + (D[j, :, :] @ D[i, :, :]).T @ Q 
            row_Q.append(blck_Q)
            blck_R = D[i, :, :].T @ R @ D[j, :, :] + D[j, :, :].T @ R @ D[i, :, :]\
                   + R @ D[j, :, :] @ D[i, :, :] + (D[j, :, :] @ D[i, :, :]).T @ R
            row_R.append(blck_R)
            
        H_Q.append(row_Q)
        H_R.append(row_R)
        
    Hess_Q = cp.bmat(H_Q)
    Hess_R = cp.bmat(H_R)
        
        
    # Define objective
    obj = cp.Minimize(t)  #+ 0.1*cp.trace(Hess_R)
    
    # Objective max of absolute value constraint
    constr = [m >=   Y @ cp.vec((Q - R) @ y) - F_s , 
              m >= -(Y @ cp.vec((Q - R) @ y) - F_s)]
    
    constr += [t >= m, m>=0]
    
    # PSD constraint
    constr += [Hess_Q >> 0, Hess_R >> 0]
    
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=True, solver = solver_name_map[solver_name]) #
    
    print(" t: ", t.value)
    print(" m: ", m.value)
    print(" error: ", (Y @ cp.vec((Q - R) @ y) - F_s ).value)
    return Q.value, R.value

def cvx_dd(D, y_s, F_s, solver_name="MOSEK"):
    """" Compute the DC decomposition of a polynomial f from samples F_s.
    
    Let F_s = f(x_s) be a sequence of samples of f evaluated at x_s and y_s = y(x_s) be
    the vector of monomials of degree up to d in the Gram form. 
     
    The function performs the DC decomposition such that 
    f = y_s'(Q-R)y where g = y'Qy, h = y'Ry are SOS convex functions. 
    
    Input: matrix of derivatives D, vector of monomials of degree up to d in the Gram form
    evaluated at sample points y_s, samples of polynomial values F_s, solver name (default
    is solver_name="MOSEK")
    
    Ouput: Gram matrices Q, R of polynomials g, h s.t. f = g - h and g = y'Qy, h = y'Ry 
    are convex. 
    """
    
    # Define problem dimension
    N = y_s.shape[1]
    N_s = y_s.shape[0]
    n = D.shape[0]
    
    # Stack data
    y = y_s.T
    Y = scipy.linalg.block_diag(*y_s)
    
    # Define optimisation variables 
    Q = cp.Variable((N, N), symmetric=True)
    R = cp.Variable((N, N), symmetric=True)
    m = cp.Variable(N_s)
    t = cp.Variable(1)
    
    # Construct Hessians
    H_Q, H_R = [], []
    for i in range(n):
        row_Q, row_R = [], []
        for j in range(n):
            blck_Q = D[i, :, :].T @ Q @ D[j, :, :] + D[j, :, :].T @ Q @ D[i, :, :]\
                   + Q @ D[j, :, :] @ D[i, :, :] + (D[j, :, :] @ D[i, :, :]).T @ Q 
            row_Q.append(blck_Q)
            blck_R = D[i, :, :].T @ R @ D[j, :, :] + D[j, :, :].T @ R @ D[i, :, :]\
                   + R @ D[j, :, :] @ D[i, :, :] + (D[j, :, :] @ D[i, :, :]).T @ R
            row_R.append(blck_R)
            
        H_Q.append(row_Q)
        H_R.append(row_R)
        
    Hess_Q = cp.bmat(H_Q)
    Hess_R = cp.bmat(H_R)
        
        
    # Define objective
    obj = cp.Minimize(t + 0.1*cp.trace(Hess_R))  #+ 0.1*cp.trace(Hess_R)
    
    # Objective max of absolute value constraint
    constr = [m >=   Y @ cp.vec((Q - R) @ y) - F_s , 
              m >= -(Y @ cp.vec((Q - R) @ y) - F_s)]
    
    constr += [t >= m, m>=0]
    
    # Diagonally dominant condition
    #constr += [ 2*cp.diag(Hess_Q) >= cp.sum(cp.abs(Hess_Q), 1), 
    #            2*cp.diag(Hess_R) >= cp.sum(cp.abs(Hess_R), 1)  ]
    n_H = Hess_Q.shape[0]
    offdiag_Hess_Q = Hess_Q - cp.diag(cp.diag(Hess_Q))
    offdiag_Hess_R = Hess_R - cp.diag(cp.diag(Hess_R))
    constr += [ cp.diag(Hess_Q) - cp.sum(cp.abs(offdiag_Hess_Q), axis=1) >= -t*np.ones(n_H), 
                cp.diag(Hess_R) - cp.sum(cp.abs(offdiag_Hess_R), axis=1) >= -t*np.ones(n_H)]
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=True, solver = solver_name_map[solver_name]) #
    
    print(" t: ", t.value)
    print(" m: ", m.value)
    print(" error: ", (Y @ cp.vec((Q - R) @ y) - F_s ).value)
    return Q.value, R.value
    
def cvx_gram(D, P, solver_name="MOSEK"):
    """" Compute the DC decomposition of a polynomial f in Gram form.
    
    Let f = y'Py where y is a vector of monomials of degree up to d, f is a polynomial of 
    degree 2d and P the Gram matrix. The function performs the DC decomposition such that 
    f = y'(Q-R)y where g = y'Qy, h = y'Ry are SOS convex functions. 
    
    Input: matrix of derivatives D, Gram matrix of polynomial P, solver name (default is
    solver_name="MOSEK")
    Ouput: Gram matrices Q, R of polynomials g, h s.t. f = g - h and g, h are convex
    """
    
    # Define problem dimension
    N = P.shape[0]
    n = D.shape[0]
    
    # Define optimisation variables 
    Q = cp.Variable((N, N), symmetric=True)
    #R = cp.Variable((N, N), symmetric=True)
    t = cp.Variable(1)
    # Construct Hessians
    H_Q, H_R = [], []
    for i in range(n):
        row_Q, row_R = [], []
        for j in range(n):
            blck_Q = D[i,:,:].T @ Q @ D[j,:,:] + D[j,:,:].T @ Q @ D[i,:,:]\
                   + Q @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ Q 
            row_Q.append(blck_Q)
            blck_R = D[i,:,:].T @ (Q-P) @ D[j,:,:] + D[j, :, :].T @ (Q-P) @ D[i,:,:]\
                   + (Q-P) @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ (Q-P)
            row_R.append(blck_R)
            
        H_Q.append(row_Q)
        H_R.append(row_R)
        
    Hess_Q = cp.bmat(H_Q)
    Hess_R = cp.bmat(H_R)
        
    n_H = Hess_Q.shape[0]  
    # Define objective
    obj = cp.Maximize(t - 0.1*cp.norm(Hess_R))  #- 0.1*cp.trace(Hess_R)
    constr = []
    # Split constraint
    
    # PSD constraint
    constr += [Hess_Q >> t*np.eye(n_H), Hess_R >> t*np.eye(n_H)]
    #constr += [Q >> t*np.eye(N), Q - P >> t*np.eye(N)]
    
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=False, solver = solver_name_map[solver_name])
    
    print("DC decomposition status: ", prob.status)
         
    return Q.value, Q.value - P

def cvx_gram_dd(D, P, solver_name="MOSEK"):
    """" Compute the DC decomposition of a polynomial f in Gram form.
    
    Let f = y'Py where y is a vector of monomials of degree up to d, f is a polynomial of 
    degree 2d and P the Gram matrix. The function performs the DC decomposition such that 
    f = y'(Q-R)y where g = y'Qy, h = y'Ry are SOS convex functions. 
    
    Input: matrix of derivatives D, Gram matrix of polynomial P, solver name (default is
    solver_name="MOSEK")
    Ouput: Gram matrices Q, R of polynomials g, h s.t. f = g - h and g, h are convex
    """
    
    # Define problem dimension
    N = P.shape[0]
    n = D.shape[0]
    
    # Define optimisation variables 
    Q = cp.Variable((N, N), symmetric=True)
    #R = cp.Variable((N, N), symmetric=True)
    t = cp.Variable(1)
    # Construct Hessians
    H_Q, H_R = [], []
    for i in range(n):
        row_Q, row_R = [], []
        for j in range(n):
            blck_Q = D[i,:,:].T @ Q @ D[j,:,:] + D[j,:,:].T @ Q @ D[i,:,:]\
                   + Q @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ Q 
            row_Q.append(blck_Q)
            blck_R = D[i,:,:].T @ (Q-P) @ D[j,:,:] + D[j, :, :].T @ (Q-P) @ D[i,:,:]\
                   + (Q-P) @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ (Q-P)
            row_R.append(blck_R)
            
        H_Q.append(row_Q)
        H_R.append(row_R)
        
    Hess_Q = cp.bmat(H_Q)
    Hess_R = cp.bmat(H_R)
        
    n_H = Hess_Q.shape[0]
    
    # Define objective
    obj = cp.Maximize(t)  #- 0.1*cp.trace(Hess_R)
    constr = []

    offdiag_Hess_Q = Hess_Q - cp.diag(cp.diag(Hess_Q))
    offdiag_Hess_R = Hess_R - cp.diag(cp.diag(Hess_R))
    constr += [ cp.diag(Hess_Q) - cp.sum(cp.abs(offdiag_Hess_Q), axis=1) >= t*np.ones(n_H), 
                cp.diag(Hess_R) - cp.sum(cp.abs(offdiag_Hess_R), axis=1) >= t*np.ones(n_H)]
    
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=False, solver = solver_name_map[solver_name])
    
    print("DC decomposition status: ", prob.status)
         
    #Q0, R0 = regularize(P, Q.value, Q.value-P)
    
    Q0, R0 = Q.value, Q.value-P

    return Q0, R0

def cvx_gram2(D, P, solver_name="MOSEK"):
    """" Compute the DC decomposition of a polynomial f in Gram form.
    
    Let f = y'Py where y is a vector of monomials of degree up to d, f is a polynomial of 
    degree 2d and P the Gram matrix. The function performs the DC decomposition such that 
    f = y'(Q-R)y where g = y'Qy, h = y'Ry are SOS convex functions. 
    
    Input: matrix of derivatives D, Gram matrix of polynomial P, solver name (default is
    solver_name="MOSEK")
    Ouput: Gram matrices Q, R of polynomials g, h s.t. f = g - h and g, h are convex
    """
    
    # Define problem dimension
    N = P.shape[0]
    n = D.shape[0]
    
    # Define optimisation variables 
    Q = cp.Variable((N, N), symmetric=True)
    #R = cp.Variable((N, N), symmetric=True)

    # Construct Hessians
    H_Q, H_R = [], []
    for i in range(n):
        row_Q, row_R = [], []
        for j in range(n):
            blck_Q = D[i,:,:].T @ Q @ D[j,:,:] + D[j,:,:].T @ Q @ D[i,:,:]\
                   + Q @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ Q 
            row_Q.append(blck_Q)
            blck_R = D[i,:,:].T @ (Q-P) @ D[j,:,:] + D[j, :, :].T @ (Q-P) @ D[i,:,:]\
                   + (Q-P) @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ (Q-P)
            row_R.append(blck_R)
            
        H_Q.append(row_Q)
        H_R.append(row_R)
        
    Hess_Q = cp.bmat(H_Q)
    Hess_R = cp.bmat(H_R)
        
    n_H = Hess_Q.shape[0]  
    # Define objective
    obj = cp.Minimize(0.1*cp.trace(Hess_R))  #- 0.1*cp.trace(Hess_R)
    constr = []
    # Split constraint
    
    # PSD constraint
    constr += [Hess_Q >> 0*np.eye(n_H), Hess_R >> 0*np.eye(n_H)]
    
    """offdiag_Hess_Q = Hess_Q - cp.diag(cp.diag(Hess_Q))
    offdiag_Hess_R = Hess_R - cp.diag(cp.diag(Hess_R))
    constr += [ cp.diag(Hess_Q) - cp.sum(cp.abs(offdiag_Hess_Q), axis=1) >= 0*np.ones(n_H), 
                cp.diag(Hess_R) - cp.sum(cp.abs(offdiag_Hess_R), axis=1) >= 0*np.ones(n_H)]"""
    
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=True, solver = solver_name_map[solver_name])
    
    print("DC decomposition status: ", prob.status)
         
    return Q.value, Q.value - P       
  

def cvx_gram2_dd(D, P, solver_name="MOSEK"):
    """" Compute the DC decomposition of a polynomial f in Gram form.
    
    Let f = y'Py where y is a vector of monomials of degree up to d, f is a polynomial of 
    degree 2d and P the Gram matrix. The function performs the DC decomposition such that 
    f = y'(Q-R)y where g = y'Qy, h = y'Ry are SOS convex functions. 
    
    Input: matrix of derivatives D, Gram matrix of polynomial P, solver name (default is
    solver_name="MOSEK")
    Ouput: Gram matrices Q, R of polynomials g, h s.t. f = g - h and g, h are convex
    """
    
    # Define problem dimension
    N = P.shape[0]
    n = D.shape[0]
    
    # Define optimisation variables 
    Q = cp.Variable((N, N), symmetric=True)
    #R = cp.Variable((N, N), symmetric=True)

    # Construct Hessians
    H_Q, H_R = [], []
    for i in range(n):
        row_Q, row_R = [], []
        for j in range(n):
            blck_Q = D[i,:,:].T @ Q @ D[j,:,:] + D[j,:,:].T @ Q @ D[i,:,:]\
                   + Q @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ Q 
            row_Q.append(blck_Q)
            blck_R = D[i,:,:].T @ (Q-P) @ D[j,:,:] + D[j, :, :].T @ (Q-P) @ D[i,:,:]\
                   + (Q-P) @ D[j,:,:] @ D[i,:,:] + (D[j,:,:] @ D[i,:,:]).T @ (Q-P)
            row_R.append(blck_R)
            
        H_Q.append(row_Q)
        H_R.append(row_R)
        
    Hess_Q = cp.bmat(H_Q)
    Hess_R = cp.bmat(H_R)
        
    n_H = Hess_Q.shape[0]  
    # Define objective
    obj = cp.Minimize(0.1*cp.trace(Hess_R) )  #- 0.1*cp.trace(Hess_R)
    constr = []
    # Split constraint
    
    # PSD constraint
    #constr += [Hess_Q >> 0*np.eye(n_H), Hess_R >> 0*np.eye(n_H)]
    
    offdiag_Hess_Q = Hess_Q - cp.diag(cp.diag(Hess_Q))
    offdiag_Hess_R = Hess_R - cp.diag(cp.diag(Hess_R))
    constr += [ cp.diag(Hess_Q) - cp.sum(cp.abs(offdiag_Hess_Q), axis=1) >= 0*np.ones(n_H), 
                cp.diag(Hess_R) - cp.sum(cp.abs(offdiag_Hess_R), axis=1) >= 0*np.ones(n_H)]
    
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=True, solver = solver_name_map[solver_name])
    
    print("DC decomposition status: ", prob.status)
         
    return Q.value, Q.value - P
           
## Hessian
def D_2(f, x_0, delta, i, j):
    """ 
    Evaluate second derivative of f along x_i and x_j at x_0:
    D_2 f = d^2 f /dx_i dx_j
    
    Input: function to differentiate f, evaluation point x_0, step delta, 
    indices of variables along which to differentiate i and j.
    Output: second order derivative along x_i and x_j
    """
    n = len(x_0)
    I = np.eye(n)
    
    return (f(x_0 + delta*I[j, :] + delta*I[i, :]) -f(x_0 + delta*I[j, :])\
    - f(x_0 + delta*I[i, :]) + f(x_0))/delta**2

def hess(f, x_0, delta):
    """
    Evaluate the Hessian of f at x_0 (numerically)
    
    Input: function whose Hessian is to be computed f, evaluation point x_0, 
    differentiation step delta. 
    Output: Hessian H. 
    """
    n = len(x_0)
    H = np.empty((n,n))
    
    for i in range(n):
        for j in range(n):
            H[i, j] = D_2(f, x_0, delta, i, j)  # compute 2nd derivative along x_i and x_j
    
    return H

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian
       
## Check split
def check(Q, R, P, f, x, expo, F_s, avg, std):
    """ A function to check the validity of a given DC decomposition
    
    f = g - h where g = y' Q y and h = y' R y are convex
    
    Will perform a series of checks to assess: 
    - if the DC decomposition describes well the original function f
    - if g, h are convex
    
    Input: the Grammians of the DC decomposition Q, R, original function f, 
    test points x, list of monomial exponents expo, function samples F_s
    
    Output: None
    """
    ## 0. Define f_, g and h from P, Q, R (note that f_ is the LS approximation of f)
    g = lambda x: monomial.polyval(expo, x, Q, avg, std)
    h = lambda x: monomial.polyval(expo, x, R, avg, std)
    f_ =lambda x: monomial.polyval(expo, x, P, avg, std)
    
    ## 1. Check f = g-h
    N = x.shape[0]  # number of test points
    
    # Generate samples
    y = monomial.eval(expo, x, avg, std)
    

    # Compute the error of least squares fit and DC decomposition
    err_LS = np.zeros(N)
    err_split = np.zeros_like(err_LS)
    #F_s = np.zeros(N)
    for i in range(N):
        #F_s[i] = f(*x[i, :])
        err_LS[i] = np.abs(g(x[i, :]) - h(x[i, :])  - F_s[i])
        #err_LS[i] = np.abs(y[i, :].T @ Q @ y[i, :] - y[i, :].T @ R @ y[i, :]  - F_s[i])
        err_split[i] = np.abs(g(x[i, :]) - h(x[i, :])  - f_(x[i, :]))
    
    print("************ Errors in LS approximation ****************")
    print("Max sample Fs: ", np.abs(F_s).max(), "/ Max absolute error: ", err_LS.max())
    print("Mean sample Fs: ", np.abs(F_s).mean(), "/ Mean absolute error: ",err_LS.mean())
    
    print("************ Error in DC decomposition ****************")
    print("Mean absolute error: ", err_split.mean())
    
    ## 2. Check convexity of g and h
    print("Checking for convexity of g and h...")
    delta = .1
    viol = 0
    for i in range(N):
        # Compute Hessians 
        H_g = hess(g, x[i, :], delta)
        H_h = hess(h, x[i, :], delta)
        
        #print("H_g eigenvalues: ", np.linalg.eigvals(H_g))
        #print("H_h eigenvalues: ", np.linalg.eigvals(H_g))
        
        # Check PSDness of Hessians (will raise 'LinAlgError' exception if not PSD)
        try: 
            scipy.linalg.cholesky(H_g)
            scipy.linalg.cholesky(H_h)
        except np.linalg.LinAlgError:
            print("Hessian not psd at iteration", i, "in x: ", x[i, :], "\n")
            viol += 1
    
    print("Checking done.")
    
    if viol == 0: print("No convexity violations.")
    else: print("{} convexity violations detected !".format(viol))

def plot(Q, R, P, f,  expo, param, avg, std):
    """ A function to visualise the validity of a given DC decomposition
    
    f = g - h where g = y' Q y and h = y' R y are convex
    
    Input: Grammians of the DC decomposition Q, R, original function f, 
    test points x, list of monomial exponents expo, parameter structure param,
    average avg and standard deviation std of training data for normalisation. 
    
    Output: None
    """
    ## 0. Define f_, g and h from P, Q, R (note that f_ is the LS approximation of f)
    g = lambda x: monomial.polyval(expo, x, Q, avg, std)
    h = lambda x: monomial.polyval(expo, x, R, avg, std)
    f_ =lambda x: monomial.polyval(expo, x, P, avg, std)
    
    
    N = 15
    tiltwing = np.linspace(param.i_w_min, param.i_w_max, 5)
    thrust = np.linspace(param.T_min, param.T_max, 5)
    Vx = np.linspace(param.v_x_min, param.v_x_max, N)
    Vz = np.linspace(param.v_z_min, param.v_z_max, N) 
    VX, VZ = np.meshgrid(Vx, Vz)
    F = np.zeros_like(VX)
    G = np.zeros_like(VX)
    H = np.zeros_like(VX)
    F_ = np.zeros_like(VX)
    err = np.zeros_like(VX)
    err_LS = np.zeros(N)
    err_split = np.zeros_like(err_LS)
    #F = np.zeros(N)
    #F_ = np.zeros(N)
    for i_w in tiltwing:
        for T in thrust:
        
            # Evaluate function f on grid
            for i in range(N):
                for j in range(N):
                    # Check for AoA
                    V = param.V(VZ[i, j], VX[i, j])
                    V_e = param.V_e(V, T)
                    alpha = i_w - np.arctan2(-VZ[i, j], VX[i, j])
                    if alpha < param.alpha_max and alpha > param.alpha_min and V_e > 0.0001:
                        F[i, j] = f(VX[i, j], VZ[i, j], i_w, T)
                        F_[i, j] = f_(np.array([VX[i, j], VZ[i, j], i_w, T]))
                        G[i, j] = g(np.array([VX[i, j], VZ[i, j], i_w, T]))
                        H[i, j] = h(np.array([VX[i, j], VZ[i, j], i_w, T]))
                        if np.abs(F[i, j]) > 0.0001:
                            err[i, j] = np.abs((F[i, j] - F_[i, j]))
                        else:
                            err[i, j] = float("nan")
                    else:
                        F[i, j] = float("nan")
                        F_[i, j] = float("nan")
                        G[i, j] = float("nan")
                        H[i, j] = float("nan")
                        err[i, j] = float("nan")
        
            
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 3, 1, projection='3d')
            ax.scatter(VX.flatten(), VZ.flatten(), F.flatten(), color='blue', label='reference')
            ax.plot_wireframe(VX, VZ, F_, rstride=1, cstride=1, color='red', label='polynomial fit')
            ax.set_xlabel('$V_x$')
            ax.set_ylabel('$V_z$')
            ax.set_zlabel('$f$')
            ax.set_title('$i_w$ = {}, $T$ = {}'.format(np.round(np.degrees(i_w), 1), np.round(T, 1)))
            ax.legend()
            
            ax = fig.add_subplot(1, 3, 3)
            cs = ax.contourf(VX, VZ, err, cmap='viridis') #locator=ticker.LogLocator()
            ax.set_xlabel('$V_x$')
            ax.set_ylabel('$V_z$')
            ax.set_title('Least-squares absolute error [$m / s^{-2}$]')
            fig.colorbar(cs)
            
            ax = fig.add_subplot(1, 3, 2, projection='3d')
            c0 = ax.plot_surface(VX, VZ, F_, alpha=0.7, linewidth=0, antialiased=True, shade=True, label='g - h')
            c1 = ax.plot_surface(VX, VZ, G, alpha=0.7, linewidth=0, antialiased=True, shade=True, label='g')
            c2 = ax.plot_surface(VX, VZ, H, alpha=0.7, linewidth=0, antialiased=True, shade=True, label='h')
            ax.set_xlabel('$V_x$')
            ax.set_ylabel('$V_z$')
            #ax.set_zlabel('$f$')
            c0._facecolors2d = c0._facecolor3d
            c0_edgecolors2d = c0._edgecolor3d
            c1._facecolors2d = c1._facecolor3d
            c1._edgecolors2d = c1._edgecolor3d
            c2._facecolors2d = c2._facecolor3d
            c2._edgecolors2d = c2._edgecolor3d
            ax.legend()
            
            fig.tight_layout()
            
            
    plt.show()
    
## DC split
def split(f, d, x_s, F_s, avg, std):
    """ 
    Compute the DC decomposition of the function f s.t. f' = g-h where f' = y' P y is the
    polynomial approximation of f of order 2d, g = y' Q y and h = y' R y are convex 
    polynomial. P, Q, R are the Gram matrices to be computed. (Note: y is the vector of 
    ordered monomials)
    
    Input: 
    - f: function whose decomposition is searched
    - p: structure of parameters
    - d: half degree of polynomial approximation
    - x_s: evaluation points
    - F_s: function values f(x_s) at evaluation points x_s
    - avg, std: mean and standard deviation of the evaluation points x_s
      
    Output: 
    - P: Gram matrix of f' =  y' P y (polynomial approximation of f of order 2d)
    - Q: Gram matrix of g = y' Q y
    - R: Gram matrix of h = y' R y
    - expo: list of exponents defining vector of monomials y
    - D: derivative matrix s.t. d/dx y(x) = D y(x)
    """
    
    n = f.__code__.co_argcount      # get the number of variables for the function f
    len_y = comb(n+d, n)            # length of monomial vector (quadratic form)
    len_z = comb(n + 2*d, n)        # length of monomial vector (canonical form)
    
    ## 1. Computation of the Gram matrix
    
    # Generate ordered exponent list for zs s.t. f = zs' @ coeff where zs is a monomial
    # vector up to degree 2d (canonical form)
    expo_full = list(monomial.expo_list(range(2*d+1), repeat=n))
    expo_full_map = dict(zip(expo_full, range(len_z)))
           
    # Generate ordered exponent list for ys s.t. f = ys' @ P @ ys where ys is a monomial
    # vector up to degree d (quadratic form)
    expo = list(monomial.expo_list(range(d+1), repeat=n)) # list of monomial exponents
    
    expo_map = dict(zip(expo, range(len_y)))  # get a hashmap (exponents -> index in expo)
    
    # Evaluate canonical monomial vector at samples
    z_s = monomial.eval(expo_full, x_s, avg, std)
    
    # Fit polynomial to data (best fit of coefficients in a least squares sense)
    coeff_f, _, _, _ = scipy.linalg.lstsq(z_s, F_s, lapack_driver='gelsy')
   
    # Other possible LS methods (do not work as well / fast)
    #coeff_f, _ = ls.least_squares_sklearn(z_s, F_s)
    #coeff_f    = ls.least_squares_cvx(z_s, F_s)
          
    # Gram matrix 
    P = monomial.gram(coeff_f, len_y, expo, expo_full, expo_full_map)
    #P = ls.least_squares_gram(y_s, F_s)
    
    ## 2. Solve DC decomposition optimisation problem
    
    # Get derivative matrices (D is derivative wrt to x_=(x-avg)/std  and D_ wrt to x)
    D, D_ = monomial.diff(n, len_y, expo, expo_map, std)
    
    # Evaluate quadratic monomial vector at samples
    y_s = monomial.eval(expo, x_s, avg, std) 
    
    # Optimisation
    Q, R = cvx_gram_dd(D, P) # works with d = 2
    #Q, R = cvx_gram(D, P)  # works with d=2, 3 / normalisation 
    #Q, R = cvx(D, y_s, F_s)  # y_s needed: uncomment y_s line
    #Q, R = cvx_dd(D, y_s, F_s)
    #Q, R = cvx_gram2(D, P)
    #Q, R = cvx_gram2_dd(D, P)
    
    return Q, R, P, expo, D, D_

    