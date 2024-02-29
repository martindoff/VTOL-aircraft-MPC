""" Functions to manipulate monomials

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
from math import comb
from scipy.linalg import block_diag, sqrtm
import time
import cvxpy as cp
import numdifftools as nd
import matplotlib.pyplot as plt 

def expo_list(*args, repeat=1):
    """ Generator for an ordered list of n-tuples representing the monomimal exponents up 
    to degree d = len(*args) 
    
    Input:
    - args: range vector of length d
    
    Output:
    - monomial_expo: generator for the monomial exponents
    """
    
    d = len(*args)
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        monomial_expo = tuple(prod)
        if (sum(monomial_expo) < d):  # only take monomials of degree up to d
            yield monomial_expo

## Derivative matrix
def diff(n, len_y, expo, expo_map, std):
    """ Construct the derivative matrix of a monomial vector y(x) s.t.
    dy/dx_i = D[i, :, :] @ y 
    
    Input:
    - n: number of variables
    - len_y: length of monomial vector
    - expo: exponent list representing the monomial vector
    - expo_map: hashmap (exponents -> index in expo)
    - std: standard deviation of training data
    
    Output: 
    - D:  derivative matrix w.r.t. normalised data x_ = (x-avg)/std
    - D_: derivative matrix w.r.t. x
    """
    D  = np.zeros((n, len_y, len_y))
    D_ = np.zeros((n, len_y, len_y))
    for i in range(n):
        for j in range(len_y):
            current_expo = expo[j] 
            diff_expo = list(current_expo)
            
            # derivate monomial wrt to state i
            if current_expo[i] > 0:
                coeff = current_expo[i]
                diff_expo[i] -= 1
                k = expo_map[tuple(diff_expo)]
                D[i, j, k]  =  coeff
                D_[i, j, k] =  coeff/std[i]
    
    return D, D_


## Monomial evaluation
def eval(expo, x_s, avg, std):
    """ Evaluate monomial vector at sample points x_s
    
    Input: 
    - expo: exponent list representing the monomial vector to be evaluated
    - x_s: sample points xs
    - avg, std: mean and standard deviation of the evaluation points x_s
    
    Output: 
    - y: monomial vector
    """
    # Normalise evaluation points
    x_s = x_s - avg
    x_s = x_s/std
    
    # Monomial vector evaluation
    exp = np.array(expo)
    if x_s.ndim > 1:
        y1 = x_s[:,None,:]**exp
        y = np.prod(y1, axis=2)
    else:
        y1 = x_s**exp
        y = np.prod(x_s**exp, axis=1)
    return y

## Polynomial evaluation
def polyval(expo, x, P, avg, std):
    """ Evaluate a polynomial defined by its Gram matrix and list of monomial exponents
    at sample points. 
    
    Input: 
    - expo: list of monomial exponents
    - x: sample points
    - P: Gram matrix
    - avg, std: mean and standard deviation of evaluation points
    """
    
    # Vector of monomials
    y = eval(expo, x, avg, std)
    
    if x.ndim > 1:
        Y = block_diag(*y)
        return Y @ (P @ y.T).flatten('F')
    else:
        return y.T @ P @ y

## Gram matrix
def gram(beta, len_y, expo, expo_full, expo_full_map):
    """ Compute Gram matrix associated with polynomial f
    
    Input: 
    - beta: polynomial coefficients in f = z @ beta where z is a vector of monomial of
    degree up to 2d (canonical form)
    - len_y: length of quadratic form monomial vector  
    
    Output: 
    - P: Gram matrix in f = y' @ P @ y where y is a vector of monomial of degree up 
    to d (quadratic form). 
    """
    N_gram = len_y  # dimension of Gram matrix
    P = np.zeros((N_gram, N_gram))
    
    # Prepare a hash map to count every time a given monomial is added
    monomial_count = dict(zip(expo_full, [0]*len(expo_full)))
    
    # Go through the Gram matrix to populate it
    for i in range(N_gram):
        for j in range(N_gram):
            monomial_expo = tuple(np.array(expo[i]) + np.array(expo[j]))  
                                                            # monomial exponent at P[i, j]
            k = expo_full_map[monomial_expo]  # position of exponent in monomial vector z
            P[i, j] = beta[k]  # store corresponding polynomial coefficient
            monomial_count[monomial_expo] += 1  # increment the scaling factor of the monomial
    # Second pass to scale the terms
    for i in range(N_gram):
        for j in range(N_gram):
            monomial_expo = tuple(np.array(expo[i]) + np.array(expo[j])) 
                                                            # monomial exponent at P[i, j]
            div = monomial_count[monomial_expo]
            P[i, j] /= div
    
    # Check if the matrix is symmetric 
    assert np.all(P == P.T), "Error: Gram matrix not symmetric"
    
    return P

# Jacobian matrix
def get_jacobian(Q1, R1, Q2, R2, D, x0, expo, avg, std):
    """ 
    Compute the Jacobian matrices Jac at x0 from the polynomial Grammians, s.t. 
    {Jac}_i = y(x0)' {Jac_gram}_i y(x0) where y(x0) is a vector of monomials evaluated at 
    x0 and {Jac_gram}_i = D_i.T @ Q + Q @ D_i and Q is the Gram matrix of the polynomial. 
     
    Input:
    - Q1, R1, Q2, R2: Gram matrices of polynomials
    - D: derivative matrix
    - x0: evaluation point
    - expo: exponent list
    - avg, std: mean and standard deviation of training data set for normalisation
    
    Output: 
    - Jac_Q1, Jac_R1, Jac_Q2, Jac_R2: Jacobian matrices at x0
    
    """
    
    
    # Problem dimension
    n = x0.shape[0]
    N = x0.shape[1] if x0.ndim > 1 else 1
        
    
    # Construct Hessians
    Jac_Q1 = np.zeros((N, n))
    Jac_R1 = np.zeros((N, n))
    Jac_Q2 = np.zeros((N, n))
    Jac_R2 = np.zeros((N, n))
    
    for j in range(n):

        Jac_Q1[:, j] = polyval(expo, x0.T, D[j, :, :].T @ Q1 + Q1 @ D[j, :, :], avg, std)
                
        Jac_R1[:, j] = polyval(expo, x0.T, D[j, :, :].T @ R1 + R1 @ D[j, :, :], avg, std)
                
        Jac_Q2[:, j] = polyval(expo, x0.T, D[j, :, :].T @ Q2 + Q2 @ D[j, :, :], avg, std)
                
        Jac_R2[:, j] = polyval(expo, x0.T, D[j, :, :].T @ R2 + R2 @ D[j, :, :], avg, std)
    
    
    return Jac_Q1, Jac_R1, Jac_Q2, Jac_R2
    
    
# Hessian matrix
def get_hessian(Q1, R1, Q2, R2, D, x0, expo, avg, std):
    """ 
    Compute the Hessian matrices Hess at x0 from the polynomial Grammians, s.t. 
    {Hess}_ij = y(x0)' {Hess_gram}_ij y(x0) where y(x0) is a vector of monomials evaluated 
    at x0 and {Hess_gram}_ij = D_i.T @ Q @ D_j + D_j.T @ Q @ D_i + Q @ D_j @ D_i 
    + (D_j @ D_i).T @ Q and Q is the Gram matrix of the polynomial. 
     
    Input:
    - Q1, R1, Q2, R2: Gram matrices of polynomials
    - D: derivative matrix
    - x0: evaluation point
    - expo: exponent list
    - avg, std: mean and standard deviation of training data set for normalisation
    
    Output: 
    - Hess_Q1, Hess_R1, Hess_Q2, Hess_R2: Hessian matrices at x0
    """
    
    # Problem dimension
    n = x0.shape[0]
    N = x0.shape[1] if x0.ndim > 1 else 1
    
    # Construct Hessians
    Hess_Q1 = np.zeros((N, n, n))
    Hess_R1 = np.zeros((N, n, n))
    Hess_Q2 = np.zeros((N, n, n))
    Hess_R2 = np.zeros((N, n, n))
    
    for i in range(n):
        for j in range(n):

            Hess_Q1[:, i, j] = polyval(expo, x0.T, D[i,:,:].T @ Q1 @ D[j,:,:]\
                       + D[j,:,:].T @ Q1 @ D[i,:,:] + Q1 @ D[j,:,:] @ D[i,:,:]\
                       + (D[j,:,:] @ D[i,:,:]).T @ Q1, avg, std)
                
            Hess_R1[:, i, j] = polyval(expo, x0.T, D[i,:,:].T @ R1 @ D[j,:,:]\
                       + D[j,:,:].T @ R1 @ D[i,:,:] + R1 @ D[j,:,:] @ D[i,:,:]\
                       + (D[j,:,:] @ D[i,:,:]).T @ R1, avg, std)
                
            Hess_Q2[:, i, j] = polyval(expo, x0.T, D[i,:,:].T @ Q2 @ D[j,:,:]\
                       + D[j,:,:].T @ Q2 @ D[i,:,:] + Q2 @ D[j,:,:] @ D[i,:,:]\
                       + (D[j,:,:] @ D[i,:,:]).T @ Q2, avg, std) 
                
            Hess_R2[:, i, j] = polyval(expo, x0.T, D[i,:,:].T @ R2 @ D[j,:,:]\
                       + D[j,:,:].T @ R2 @ D[i,:,:] + R2 @ D[j,:,:] @ D[i,:,:]\
                       + (D[j,:,:] @ D[i,:,:]).T @ R2, avg, std) 
    
    
    return Hess_Q1, Hess_R1, Hess_Q2, Hess_R2
    
# Quadratic approximation
def quad_approx(Q1, R1, Q2, R2, D, z_0, expo, param, avg, std):
    """" Obtain quadratic approximations of polynomials g1, h1, g2, h2 defined from their
    Grammians Q1, R1, Q2, R2 respectively, evaluated at z_0 s.t. 
    
    f_quad(z) = f(z_0) + J (z - z_0) + (z - z_0)' H (z - z_0)
    
    where f_quad is the quadratic approximation of f at z_0, J is the Jacobian and H the 
    Hessian, z = [x; u]
    
    Input:
    - Q1, R1, Q2, R2: Grammians
    - D: derivative matrix
    - z_0: evaluation trajectory
    - expo: exponent list
    - param: structure of parameters
    - avg, std: average and standard deviation of training data
    
    Output:
    - g1_quad, h1_quad, g2_quad, h2_quad: quadratic approximations evaluated at z_0
    """
    
    # Evaluate polynomials
    g1 = polyval(expo, z_0.T, Q1, avg, std)
    h1 = polyval(expo, z_0.T, R1, avg, std)
    g2 = polyval(expo, z_0.T, Q2, avg, std)
    h2 = polyval(expo, z_0.T, R2, avg, std)
    
    
    # Compute Jacobians
    Jac_Q1, Jac_R1, Jac_Q2, Jac_R2 = get_jacobian(Q1, R1, Q2, R2, D, z_0, expo, avg, std)
    
    # Compute Hessians
    Hess_Q1, Hess_R1, Hess_Q2, Hess_R2 = get_hessian(Q1, R1, Q2, R2, D, z_0, expo, avg, std)
    
    # Square root Hessians
    N = Hess_Q1.shape[0]
    sqrt_Q1 = np.zeros_like(Hess_R1)
    sqrt_R1 = np.zeros_like(Hess_R1)
    sqrt_Q2 = np.zeros_like(Hess_R1)
    sqrt_R2 = np.zeros_like(Hess_R1)
    for l in range(N):
        sqrt_Q1[l, :, :] = sqrtm(Hess_Q1[l, :, :])
        sqrt_R1[l, :, :] = sqrtm(Hess_R1[l, :, :])
        sqrt_Q2[l, :, :] = sqrtm(Hess_Q2[l, :, :])
        sqrt_R2[l, :, :] = sqrtm(Hess_R2[l, :, :])
    
    # Define quadratic approximations
    sqrt_Q1_ = block_diag(*sqrt_Q1)
    sqrt_R1_ = block_diag(*sqrt_R1)
    sqrt_Q2_ = block_diag(*sqrt_Q2)
    sqrt_R2_ = block_diag(*sqrt_R2)
    
    Jac_Q1_ = block_diag(*Jac_Q1)
    Jac_R1_ = block_diag(*Jac_R1)
    Jac_Q2_ = block_diag(*Jac_Q2)
    Jac_R2_ = block_diag(*Jac_R2)
    
    if z_0.ndim > 1:
        #print(Jac_R1 * z_0.shape, z_0.shape)
        g1_quad = lambda z: cp.reshape(Jac_Q1_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[1],))\
        +cp.square(cp.norm(cp.reshape(sqrt_Q1_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[0], z_0.shape[1])), axis=0))
        h1_quad = lambda z: cp.reshape(Jac_R1_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[1],))\
        +cp.square(cp.norm(cp.reshape(sqrt_R1_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[0], z_0.shape[1])), axis=0))
        g2_quad = lambda z: cp.reshape(Jac_Q2_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[1],))\
        +cp.square(cp.norm(cp.reshape(sqrt_Q2_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[0], z_0.shape[1])), axis=0))
        h2_quad = lambda z: cp.reshape(Jac_R2_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[1],))\
        +cp.square(cp.norm(cp.reshape(sqrt_R2_ @\
         cp.reshape(z-z_0, (z_0.shape[0]*z_0.shape[1],1)), (z_0.shape[0], z_0.shape[1])), axis=0))
    else:
        g1_quad = lambda z: Jac_Q1 @ (z - z_0) + cp.square(cp.norm(cp.reshape(sqrt_Q1_ @\
         cp.reshape(z-z_0, (z_0.shape[0],1)), (z_0.shape[0],1)), axis=0))[:, None]
        h1_quad = lambda z: Jac_R1 @ (z - z_0) + cp.square(cp.norm(cp.reshape(sqrt_R1_ @\
         cp.reshape(z-z_0, (z_0.shape[0],1)), (z_0.shape[0],1)), axis=0))[:, None]
        g2_quad = lambda z: Jac_Q2 @ (z - z_0) + cp.square(cp.norm(cp.reshape(sqrt_Q2_ @\
         cp.reshape(z-z_0, (z_0.shape[0],1)), (z_0.shape[0],1)), axis=0))[:, None]
        h2_quad = lambda z: Jac_R2 @ (z - z_0) + cp.square(cp.norm(cp.reshape(sqrt_R2_ @\
         cp.reshape(z-z_0, (z_0.shape[0],1)), (z_0.shape[0],1)), axis=0))[:, None]
         
    #print("look compat:", sqrt_Q1.shape, z_0.T.shape, (sqrt_Q1 @ z_0.T[:,:, None]).shape)
    return g1_quad, h1_quad, g2_quad, h2_quad, Hess_Q1, Hess_R1, Hess_Q2, Hess_R2

def plot(Q1, R1, Q2, R2, D, z_0, expo, param, avg, std):
    """ A function to visualise the validity of a quadratic approximation of a function f
    
    Input: original function f, quadratic approximation f_quad, parameter structure param,
    average avg and standard deviation std of training data for normalisation. 

    Output: None
    """
    
    # Initialise variables
    N = 15
    Vx = np.linspace(param.v_x_min, param.v_x_max, N)
    Vz = np.linspace(param.v_z_min, param.v_z_max, N) 
    VX, VZ = np.meshgrid(Vx, Vz)
    
    # Define polynomials
    g1 = lambda x: polyval(expo, x, Q1, avg, std)
    h1 = lambda x: polyval(expo, x, R1, avg, std)
    g2 = lambda x: polyval(expo, x, Q2, avg, std)
    h2 = lambda x: polyval(expo, x, R2, avg, std)
    
    # Pick 5 randomly selected evaluation points for approximation
    for l in range(10):
        rand_index = np.random.randint(0, N)
        
        # Zoom on random point
        v_x_rand = z_0[0, rand_index]
        v_z_rand = z_0[1, rand_index]
        offset = 3
        Vx = np.linspace(v_x_rand-offset, v_x_rand+offset, N)
        Vz = np.linspace(v_z_rand-offset, v_z_rand+offset, N) 
        VX, VZ = np.meshgrid(Vx, Vz)
            
        # Quadratic approximation
        g1_, h1_, g2_, h2_, _, _, _, _ = quad_approx(Q1, R1, Q2, R2, 
                                             D, z_0[:, rand_index], expo, param, avg, std)
                                             
        # Init 
        G1 = np.zeros_like(VX)
        H1 = np.zeros_like(VX)
        G2 = np.zeros_like(VX)
        H2 = np.zeros_like(VX)
        G1_ = np.zeros_like(VX)
        H1_ = np.zeros_like(VX)
        G2_ = np.zeros_like(VX)
        H2_ = np.zeros_like(VX)
        
        err = np.zeros_like(VX)
        err_LS = np.zeros(N)
        err_split = np.zeros_like(err_LS)
            
        i_w = z_0[2, rand_index]
        T = z_0[3, rand_index]
        
        # Evaluate function f on grid
        for i in range(N):
            for j in range(N):
                z = np.array([VX[i, j], VZ[i, j], i_w, T])
                # nonlinear functions 
                G1[i, j] = g1(z)
                H1[i, j] = h1(z)
                G2[i, j] = g2(z)
                H2[i, j] = h2(z)
                            
                # quadratic approx
                #print(g1_(z), g1_(z).shape)
                G1_[i, j] = g1(z_0[:, rand_index]) + g1_(z).value
                H1_[i, j] = h1(z_0[:, rand_index]) + h1_(z).value
                G2_[i, j] = g2(z_0[:, rand_index]) + g2_(z).value
                H2_[i, j] = h2(z_0[:, rand_index]) + h2_(z).value
          
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        # G1
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(z_0[0, rand_index], z_0[1, rand_index], g1(z_0[:, rand_index]), 
                                            linewidths=2, marker= "X", label='eval point')
        ax.scatter(VX.flatten(), VZ.flatten(), G1.flatten(), label='ref')
        #ax.plot_wireframe(VX, VZ, F_, rstride=1, cstride=1, color='red', label='quadratic fit')
        c0 = ax.plot_surface(VX, VZ, G1_, alpha=0.7, linewidth=0, antialiased=True, 
                                                        shade=True, label='quadratic fit')
        c0._facecolors2d = c0._facecolor3d
        c0_edgecolors2d = c0._edgecolor3d
        #ax.plot_surface(VX, VZ, G, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        #ax.plot_surface(VX, VZ, H, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        ax.set_xlabel('$V_x$')
        ax.set_ylabel('$V_z$')
        ax.set_zlabel('$g1$')
        ax.set_title('$i_w$ = {}, $T$ = {}'.format(np.round(np.degrees(i_w), 1), np.round(T, 1)))
        ax.legend()
                
        # H1        
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter(z_0[0, rand_index], z_0[1, rand_index], h1(z_0[:, rand_index]), 
                                            linewidths=2, marker= "X", label='eval point')
        ax.scatter(VX.flatten(), VZ.flatten(), H1.flatten(), label='ref')
        #ax.plot_wireframe(VX, VZ, F_, rstride=1, cstride=1, color='red', label='quadratic fit')
        c0 = ax.plot_surface(VX, VZ, H1_, alpha=0.7, linewidth=0, antialiased=True, 
                                                        shade=True, label='quadratic fit')
        c0._facecolors2d = c0._facecolor3d
        c0_edgecolors2d = c0._edgecolor3d
        #ax.plot_surface(VX, VZ, G, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        #ax.plot_surface(VX, VZ, H, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        ax.set_xlabel('$V_x$')
        ax.set_ylabel('$V_z$')
        ax.set_zlabel('$h1$')
        #ax.set_title('$i_w$ = {}, $T$ = {}'.format(np.round(np.degrees(i_w), 1), np.round(T, 1)))
        ax.legend()
         
        # G2       
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(z_0[0, rand_index], z_0[1, rand_index], g2(z_0[:, rand_index]), 
                                            linewidths=2, marker= "X", label='eval point')
        ax.scatter(VX.flatten(), VZ.flatten(), G2.flatten(), label='ref')
        #ax.plot_wireframe(VX, VZ, F_, rstride=1, cstride=1, color='red', label='quadratic fit')
        c0 = ax.plot_surface(VX, VZ, G2_, alpha=0.7, linewidth=0, antialiased=True, 
                                                        shade=True, label='quadratic fit')
        c0._facecolors2d = c0._facecolor3d
        c0_edgecolors2d = c0._edgecolor3d
        #ax.plot_surface(VX, VZ, G, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        #ax.plot_surface(VX, VZ, H, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        ax.set_xlabel('$V_x$')
        ax.set_ylabel('$V_z$')
        ax.set_zlabel('$g2$')
        #ax.set_title('$i_w$ = {}, $T$ = {}'.format(np.round(np.degrees(i_w), 1), np.round(T, 1)))
        ax.legend()
        
        # H2      
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(z_0[0, rand_index], z_0[1, rand_index], h2(z_0[:, rand_index]), 
                                            linewidths=2, marker= "X", label='eval point')
        ax.scatter(VX.flatten(), VZ.flatten(), H2.flatten(), label='ref')
        #ax.plot_wireframe(VX, VZ, F_, rstride=1, cstride=1, color='red', label='quadratic fit')
        c0 = ax.plot_surface(VX, VZ, H2_, alpha=0.7, linewidth=0, antialiased=True, 
                                                        shade=True, label='quadratic fit')
        c0._facecolors2d = c0._facecolor3d
        c0_edgecolors2d = c0._edgecolor3d
        #ax.plot_surface(VX, VZ, G, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        #ax.plot_surface(VX, VZ, H, alpha=0.7, linewidth=0, antialiased=True, shade=True)
        ax.set_xlabel('$V_x$')
        ax.set_ylabel('$V_z$')
        ax.set_zlabel('$h2$')
        #ax.set_title('$i_w$ = {}, $T$ = {}'.format(np.round(np.degrees(i_w), 1), np.round(T, 1)))
        ax.legend()
         
        fig.tight_layout()
            
    plt.show()
    
    