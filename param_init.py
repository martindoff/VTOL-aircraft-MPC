""" Model parameters of the VTOL aircraft

(c) 03/2023 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
from wing import Wing
import math

## Vehicle parameters
m = 752.2                                     # Aircraft mass (kg)
rho = 1.225                                   # Air density (kg/m^3)
S = 8.93                                      # Area of the wing (m2)
A = np.pi*0.95**2                             # Rotor surface (m2)
n_mot=4;                                      # Number of motors (-) 
S_star = S/(n_mot*A)                          # Ratio wing / rotor area (-) 
g = 9.81                                      # Gravity acceleration (m/s^2)
mu = 0.73                                     # Portion of the wing in slipstream (-)
J_w = 1/12*(330*4*10)*1**2                    # Wing moment of inertia (kg.m2)
AR = 5.29                                     # Aspect ratio (-)
e = 1.3                                       # Oswald number (-)
C_D_0 = 0.012                                 # Zero-lift drag coefficient (-)
t_c = 12/100                                  # Thickness to chord ratio (-)
b_0, b_1 = 0, 0.076*180.0/np.pi               # Lift coefficients (-, 1/rad)
alpha_s = math.radians(15)                    # Angle of attack at stall (rad)
v_stall = np.sqrt(m*g/\
          (0.5*rho*S*math.radians(3)*b_1))    # Stall speed (ms/s)

## Wind gust parameters
W_low, W_up = 0, 0

## Wing aerodynamic parameters
wing_VTOL = Wing(AR, e, C_D_0, t_c,
                 b_0, b_1, alpha_s) 

## State and input constraints
T_min = 0                                     # Minimum thrust (N)
T_max = m*g*1.2                               # Maximum thrust (N)
alpha_min = math.radians(-90)                   # Minimum angle of attack (rad) 
alpha_max = math.radians(90)                  # Maximum angle of attack (rad)     
gamma_min = math.radians(-90)                 # Minimum flight path angle (rad) 
gamma_max = math.radians(90)                  # Maximum flight path angle (rad)
i_w_min = math.radians(0)                   # Minimum tiltwing angle (rad) 
i_w_max = math.radians(90)                   # Maximum tiltwing angle (rad)
v_x_min = 0                                   # Minimum forward velocity (m/s)
v_x_max = 60                                   # Maximum forward velocity (m/s) 
v_z_min = -20  /np.sqrt(2)                                 # Minimum vertical velocity (m/s)
v_z_max = 20/np.sqrt(2)                                    # Maximum vertical velocity (m/s)
a_min = -.3*g                                 # Minimum acceleration (m/s^2)
a_max = .3*g                                  # Maximum acceleration (m/s^2)
M_min = -50                                   # Minimum wing momentum (N.m)
M_max = 50                                    # Maximum wing momentum (N.m)
ctr =[[v_x_min, v_x_max], 
      [v_z_min, v_z_max], 
      [i_w_min, i_w_max], 
      [T_min, T_max]]
ctr = np.array(ctr) 
x_max = np.array([v_x_max, 
                  v_z_max]) 
x_min = np.array([v_x_min, 
                  v_z_min])
u_max = np.array([i_w_max, T_max]) 
u_min = np.array([i_w_min, T_min])

## Nonlinear dynamics
# Intermediate state definition
V_e = lambda V, T : np.sqrt(V**2 + 2*T/(n_mot*A*rho))  # Effective velocity
alpha_e = lambda AoA, V, T : np.arcsin(np.sin(AoA)*V/V_e(V, T))  # Effective AoA
gamma = lambda Vz, Vx: np.arctan2(-Vz, Vx) 
V = lambda Vz, Vx: np.sqrt(Vz**2 + Vx**2)

# V_x dynamics
f1 = lambda Vx, Vz, i_w, T: (T*np.cos(i_w) \
- mu*0.5*rho*S*wing_VTOL.C_L(alpha_e(i_w - gamma(Vz, Vx), V(Vz, Vx), T))\
*V_e(V(Vz, Vx), T)**2*np.sin(gamma(Vz, Vx))\
- mu*0.5*rho*S*wing_VTOL.C_D(alpha_e(i_w - gamma(Vz, Vx), V(Vz, Vx), T))\
*V_e(V(Vz, Vx), T)**2*np.cos(gamma(Vz, Vx))\
- (1-mu)*0.5*rho*S*wing_VTOL.C_L(i_w - gamma(Vz, Vx))*V(Vz, Vx)**2*np.sin(gamma(Vz, Vx))\
- (1-mu)*0.5*rho*S*wing_VTOL.C_D(i_w - gamma(Vz, Vx))*V(Vz, Vx)**2*np.cos(gamma(Vz, Vx)))/m

# V_z dynamics 
f2 = lambda Vx, Vz, i_w, T: -(T*np.sin(i_w) - m*g \
+ mu*0.5*rho*S*wing_VTOL.C_L(alpha_e(i_w - gamma(Vz, Vx), V(Vz, Vx), T))\
*V_e(V(Vz, Vx), T)**2*np.cos(gamma(Vz, Vx))\
- mu*0.5*rho*S*wing_VTOL.C_D(alpha_e(i_w - gamma(Vz, Vx), V(Vz, Vx), T))\
*V_e(V(Vz, Vx), T)**2*np.sin(gamma(Vz, Vx))\
+ (1-mu)*0.5*rho*S*wing_VTOL.C_L(i_w - gamma(Vz, Vx))*V(Vz, Vx)**2*np.cos(gamma(Vz, Vx))\
- (1-mu)*0.5*rho*S*wing_VTOL.C_D(i_w - gamma(Vz, Vx))*V(Vz, Vx)**2*np.sin(gamma(Vz, Vx)))/m

##########################################################################################
################################ Forward transition ######################################
##########################################################################################

## Initial state
V_x_init = 0.1                                  # Initial forward velocity (m/s)
V_z_init = 0                                    # Initial vertical velocity (m/s)
x_init = np.array([V_x_init, V_z_init])         # Initial state

## Initial input 
i_w_init = math.radians(75)                     # Initial tiltwing angle (rad)                                     # Initial wing torque (Nm)
T_pot = m*g/np.sin(i_w_init)                
T_init = T_pot if T_pot < T_max else T_max      # Initial thrust (N)
u_init = np.array([i_w_init, T_init])           # Initial input

## Reference state
V_x_r = 52                                      # Reference forward velocity (m/s)
V_z_r = 0                                       # Reference vertical velocity (m/s)
h_r = np.array([V_x_r, V_z_r])                  # Reference state
Q = np.diag([1, 1])                             # State penalty matrix
          
## Reference input
T_r = 624                                       # Reference thrust (N)
i_w_r = math.radians(0)                         # Reference tiltwing angle (rad)
u_r = np.array([i_w_r, T_r])
R = np.diag([.1, .1])/60                        # Input penalty matrix
 
## Terminal set parameters
u_term = np.array([math.radians(3), 100])       # Terminal set bound on input 
x_term = np.array([6, 5])                       # Terminal set bound on state 