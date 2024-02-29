""" Collection of algorithms to solve a least squares problem

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import scipy.linalg
import scipy.optimize
import cvxpy as cp
from sklearn.linear_model import LinearRegression

## Backward substitution
def backward(R, y):
    """ 
    Implement backward substitution algorithm to solve linear system of equations:
    
    R x = y
    
    for x where R is upper triangular. 
    """
    
    n = R.shape[0]
    m = R.shape[1]
    x = np.zeros(m)
    for i in range(n):
        s = R[-1-i, :] @ x
        x[-1-i] = (y[-1-i] - s)/R[-1-i, -1-i]
    
    return x

## Solve normal equation
def least_squares_normal(A, b):
    """ 
    Fit polynomial to data by solving the least squares normal equation A'Ax = A'b for x. 
    This is done by performing a QR decomposition of the A matrix, followed by 
    backsubstitution.
    
    Input:  matrix whose rows are the vectors of monomials evaluated at samples A, 
    polynomial values evaluated at sample b
    Output: coefficients beta in f = z @ beta such that the model is a best fit to the
    data in a least squares sense. 
    """
    
    # 1. QR decomposition of z_s:
    Q, R = scipy.linalg.qr(A)  # normal equation reduces to: R @ beta = Q.T @ F_s 
    y = Q.T @ b
    
    # 2. Backsubstitution
    beta = backward(R, y)
    
    return beta

## LS with sklearn       
def least_squares_sklearn(A, b):
    """ 
    Fit polynomial to data by solving the least squares problem with sklearn. 
    
    Input:  matrix whose rows are the vectors of monomials evaluated at samples A, 
    polynomial values evaluated at sample b
    Output: coefficients beta in f = z @ beta such that the model is a best fit to the
    data in a least squares sense. 
    """
    
    model = LinearRegression()  # model creation
    model.fit(A, b)             # train model
    return model.coef_, model.score(A, b)

## L1 norm least squares
def least_squares_abs(z_s, F_s):
    """ L1 norm least squares - fit polynomial to data by solving the least squares 
    problem with linprog 
    
    Input:  matrix whose rows are the vectors of monomials evaluated at samples z_s, 
    polynomial values evaluated at sample F_s
    Output: coefficients beta in f = z @ beta such that the model is a best fit to the
    data in a least squares sense. 
    """
    
    N_s = z_s.shape[0]  # number of samples
    N = z_s.shape[1]    # number of polynomial coefficients

    # Objective  
    c = np.vstack([np.zeros((N, 1)), np.ones((N_s, 1))])
    
    # Constraint 
    I = np.eye(N_s)
    b = np.vstack([F_s[:, None], -F_s[:, None], np.zeros((N_s, 1))])
    A1 = np.hstack([z_s, -I])
    A2 = np.hstack([-z_s, -I])
    A3 = np.hstack([np.zeros((N_s, N)), -I])
    A = np.vstack([A1, A2, A3])

    # Solve
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b)
    print(res.message)
    print("Optimal value: ", res.fun)
    
    # Extract solution
    beta = res.x[0:N]
    
    return beta
    
## LS with cvx
def least_squares_cvx(y_s, F_s, solver_name = "MOSEK"):
    """" Solve a least square problem with CVX
    
    Fit polynomial coefficients beta s.t. model y' beta is a best fit to the N_s samples 
    F_s = f(x_s). 
    
    Input: vector of monomials of degree up to 2d evaluated at sample points x_s, function
    evaluation at sample points F_s = f(x_s), solver name (default is solver_name="MOSEK")
    Output: coefficients beta in model y' beta
    """
    
    # Define problem dimension
    N = y_s.shape[1]
    N_s = y_s.shape[0]
    
    # Define optimisation variables 
    beta = cp.Variable(N)
    t = cp.Variable(N_s)
        
        
    # Define objective
    obj = cp.Minimize(cp.sum(t)) 
    
    # Constr
    constr = [t >=   y_s @ beta - F_s , 
              t >= -(y_s @ beta - F_s), t>=0]
    # Assemble problem
    prob = cp.Problem(obj, constr) # optimisation problem 
        
    # Solve problem
    solver_name_map = {"ECOS": cp.ECOS, "MOSEK": cp.MOSEK, "OSQP": cp.OSQP, "SCS": cp.SCS}
    prob.solve(verbose=False, solver = solver_name_map[solver_name]) #
    
    return beta.value

## Constrained LS in the Gram form
def least_squares_gram(Y, F_s):
    """ Solve constrained least square problem in the Gram form:
    
    min_P sum_s (ys^T @ P @ ys - F_s)^2
    
    s.t. P^T = P
    
    Input: matrix Y containing the monomial vectors y_s, samples of function evaluation F_s
    Output: Gram matrix P that is a best fit to the data. 
    """
    
    N = Y.shape[1]
    N_s = Y.shape[0]
    N_vech = N*(N+1)//2  # number of elements in the half vectorisation
    b = F_s/2
    A = np.empty((N_s, N_vech))
    iu = np.triu_indices(N)
    
    for i in range(N_s):
        y_tri = Y[i, :, None] @ Y[i, :, None].T
        row = y_tri[iu]  # half vectorize
        A[i, :] = row
        
    U_vech, _, _, _ = scipy.linalg.lstsq(A, b, lapack_driver='gelsy')
    
    # Transform U back in triangular form
    U = np.zeros((N, N))
    k = 0
    for i in range(N):
        for j in range(N):
            if j >= i:
                U[i, j] = U_vech[k]
                k += 1
    
    # Compute Gram matrix
    P = U + U.T
    
    return P