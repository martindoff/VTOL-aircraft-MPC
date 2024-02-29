""" Wing data

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import math

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

class Wing():
    """ 
    Class to implement the aerodynamics of a wing 
    
    """
    
    ## Constructor
    def __init__(self, AR, e, C_D_0, t_c, b_0, b_1, alpha_s):
        """
        Initialise wing model
        
        """
        # Basic wing parameters
        self.AR = AR
        self.e = e
        self.C_D_0 = C_D_0
        self.t_c = t_c
        
        # Linear lift coefficients
        self.b_0 = b_0
        self.b_1 = b_1
        self.alpha_s = alpha_s
        
        # Post stall lift model
        C_L_s = b_0 + b_1*alpha_s
        C_1 = 1.1 + 0.018*AR
        self.A_1 = C_1/2
        self.A_2 = (C_L_s - C_1*np.sin(alpha_s)*np.cos(alpha_s))*np.sin(alpha_s)\
                   /(np.cos(alpha_s))**2
                   
        # Quadratic drag coefficients 
        self.a_2 = b_1**2/(e*np.pi*AR)
        self.a_1 = 2*b_1*b_0/(e*np.pi*AR)
        self.a_0 = (b_0**2/(e*np.pi*AR)+C_D_0) # drag coefficients (for angles in rad)
        
        # Post stall drag model
        C_D_s = self.a_0 + self.a_1*alpha_s + self.a_2*alpha_s**2
        self.C_D_max = (1 + 0.065*AR)/(0.9 + t_c)
        self.B_2 = (C_D_s - self.C_D_max*np.sin(alpha_s))/np.cos(alpha_s)
    
    ## C_L and C_D functions 
    def C_L(self, AoA):
        """ Compute lift coefficient
        
        Input: effective angle of attack AoA (rad)
        Output: lift coefficient (-)
        """
        Coeff_L   = lambda x : self.b_0 + self.b_1*x  # linear lift model 
        Coeff_L_s = lambda x : self.A_1*np.sin(2*x)\
                             + self.A_2*np.cos(x)**2/np.sin(x)  # post-stall model
        if isinstance(AoA, Iterable):
            return np.array([Coeff_L(x) if x < self.alpha_s else Coeff_L_s(x) for x in AoA])
        else:
            return Coeff_L(AoA) if AoA < self.alpha_s else Coeff_L_s(AoA)
    
    def C_D(self, AoA):
        """ Compute drag coefficient
        
        Input: effective angle of attack AoA (rad)
        Output: drag coefficient (-)
        """
        Coeff_D   = lambda x : self.a_0 + self.a_1*x + self.a_2*x**2  # quad. drag model 
        Coeff_D_s = lambda x : self.C_D_max*np.sin(x) + self.B_2*np.cos(x)  #  AoA>20 deg
        
        if isinstance(AoA, Iterable):
            return np.array([Coeff_D(x) if x < 15*np.pi/180 else Coeff_D_s(x) for x in AoA])
        else:
            return Coeff_D(AoA) if AoA < 15*np.pi/180 else Coeff_D_s(AoA)
    
    ## Lift and drag
    def L(V, AoA, T, self):
        """ Compute lift force
        
        Input: velocity V (m/s), effective angle of attack AoA (rad), thrust T (N)
        Output: lift force (N)
        """
        return 0
        
    def D(V, AoA, T, self):
        """ Compute drag force
        
        Input: velocity V (m/s), effective angle of attack AoA (rad), thrust T (N)
        Output: drag force (N)
        """
        return 0

##########################################################################################
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib
    import os
    from param_init import AR, e, C_D_0, t_c, b_0, b_1, alpha_s
    from scipy.interpolate import CubicSpline
    
    # Wing
    AoA = np.linspace(0, np.pi/2, 100)
    AoA2 = np.linspace(0, np.pi/2, 5)    
    wing_VTOL = Wing(AR, e, C_D_0, t_c, b_0, b_1, alpha_s)
    cl = CubicSpline(AoA2, wing_VTOL.C_L(AoA2))
    cd = CubicSpline(AoA2, wing_VTOL.C_D(AoA2))
    
    #wing_VTOL.C_L = cl
    #wing_VTOL.C_D = cd
    
    # Plot
    if not os.path.isdir('plot'):
        os.mkdir('plot')
       
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['axes.unicode_minus'] = False
    def math_formatter(x, pos):
        return "${}$".format(x).replace("-", u"\u2212")
        
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.degrees(AoA), wing_VTOL.C_L(AoA), '-')
    plt.plot(np.degrees(AoA), cl(AoA), '-')
    plt.title('$C_L$ (-)')
    plt.xlabel(r'$\alpha$, $\alpha_e$ (deg)')
    plt.grid()


    plt.subplot(1, 2, 2)
    plt.plot(np.degrees(AoA), wing_VTOL.C_D(AoA), '-')
    plt.plot(np.degrees(AoA), cd(AoA), '-')
    plt.title('$C_D$ (-)')
    plt.xlabel(r'$\alpha$, $\alpha_e$ (deg)')
    plt.grid()

    plt.savefig('plot/coef.eps', format='eps') 
    plt.savefig('plot/coef.pdf', format='pdf') 
    
    plt.show()

