# VTOL-aircraft-MPC
Data-driven Robust Model Predictive Control of Tiltwing Vertical Take-Off and Landing Aircraft

## Background 

Simulation results for the forward transition of a VTOL aircraft using data-driven robust nonlinear MPC as presented in the paper "Data-driven Robust Model Predictive Control of Tiltwing Vertical Take-Off and Landing Aircraft" by M. Doff-Sotta, M. Cannon, M. Bacic. The controller leverages techniques from difference of convex (DC) decomposition of polynomials to represent the VTOL nonlinear dynamics in DC form, which allows to use the [DC-TMPC algorithm](https://github.com/martindoff/DC-TMPC) presented in the [paper](https://ora.ox.ac.uk/objects/uuid:a3a0130b-5387-44b3-97ae-1c9795b91a42/download_file?safe_filename=Doff-Sotta_and_Cannon_2022_Difference_of_convex.pdf&file_format=application%2Fpdf&type_of_work=Conference+item): the DC dynamics is linearized successively and we derive bounds on the necessarily convex linearization error. By convexity, these bounds are tights and the linearization error can be treated as a bounded disturbance in a robust MPC framework. The resulting controller is computationally tractable and can be applied to systems defined from data. We consider here a case study from urban air mobility.  

## Built with

* Python 3
* CVXPY


## Getting started

1. Clone the repository
   ```sh
   git clone https://github.com/martindoff/VTOL-aircraft-MPC.git
   ```
2. Go to folder

   ```sh
   cd VTOL-aircraft-MPC
   ```
   
3. Run code

   ```sh
   python3 forward_main.py
   ```
