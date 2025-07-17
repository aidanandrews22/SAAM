import os
import numpy as np
import math
from scipy.linalg import solve_continuous_are

def find_nearest_upright(theta):
    return np.pi * 2 * round(theta / (2 * np.pi))
    # return np.pi * (2 * round((theta - np.pi) / (2 * np.pi)) + 1)

def LQR_controller(state, cartmass, polemass, polelength):
    """
    state: [x, x_dot, theta, theta_dot]
    """
    x, x_dot, theta, theta_dot = state
    g = 9.81

    eq_theta = find_nearest_upright(theta)
    eq_pt = np.array([0.0, 0.0, eq_theta, 0.0])

    total_mass = cartmass + polemass
    alpha = polelength * (4 * polemass + cartmass) / (3 * total_mass)

    a23 = - (g * polemass * polelength) / (total_mass * alpha)
    a43 = g / alpha

    b2 = (1/total_mass) + (polemass * polelength) / (total_mass**2 * alpha)
    b4 = - 1 / (total_mass * alpha)

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, a23, 0],
        [0, 0, 0, 1],
        [0, 0, a43, 0]
    ])

    B = np.array([
        [0],
        [b2],
        [0],
        [b4]
    ])

    Q = np.diag([1, 1, 1, 1])
    R = np.array([[1]])


    # Solve the continuous-time algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    # Compute the LQR gain
    K = np.linalg.inv(R) @ B.T @ P
    K = K.flatten()  # Ensure K is a 1D array

    return K



def swingup_lqr_controller(state, switched, cartmass, polemass, polelength):
# def swingup_lqr_controller(state, cartmass, polemass, polelength):
    """
    state: [x, x_dot, theta, theta_dot]
    switched: boolean indicating if the controller has switched to LQR (bool)
    """
    x, x_dot, theta, theta_dot = state
    g = 9.81

    eq_theta = find_nearest_upright(theta)
    eq_pt = np.array([0.0, 0.0, eq_theta, 0.0])
    # print(f"eq_theta: {eq_theta}")
    # eq_pt = np.array([0.0, 0.0, 0.0, 0.0])  # Upright position
    # eq_theta = 0.0  # Upright position
    

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Switching Conditions for LQR
    if switched or (np.abs(theta- eq_theta) < 0.3 and np.abs(theta_dot) < 1):
        # LQR fcn here later
        # if switched == False:
            # print("Switching to LQR controller")
            
        switched = True
        K_lqr = LQR_controller(state, cartmass, polemass, polelength)
        f = - K_lqr @ (state - eq_pt)
    else:
        # Energy-based controller
        K_e = 1.0 # 0.4
        K_p = 1.0
        K_d = 1.0

        # Etilde = 0.5 * polemass * polelength**2 * theta_dot**2 + polemass * g * polelength * (cos_theta)

        ### works with larger switching conditions, not the best controller
        # E = 0.5 * polemass * polelength**2 * theta_dot**2 + polemass * g * polelength * (1 + cos_theta)
        # E_target = 2 * polemass * g * polelength
        # Etilde = E - E_target
        ### 

        theta1 = theta + np.pi
        cos_theta1 = np.cos(theta1)
        sin_theta1 = np.sin(theta1)
        theta1_dot = -theta_dot

        Etilde = -g*polelength*polemass + (1/2)*polelength*polemass*(-2*g*cos_theta1 + polelength*theta1_dot*theta1_dot)
        xpp_desired =  K_e*cos_theta1*theta1_dot*Etilde - K_p*x - K_d*x_dot
        theta1_pp = -cos_theta1/polelength * xpp_desired - g/polelength * sin_theta1
        f = (polemass+cartmass)*xpp_desired + cos_theta1*polelength*polemass*theta1_pp - sin_theta1*polelength*polemass*theta1_dot*theta1_dot


        ###
        # E = 0.5 * polemass * polelength**2 * theta_dot**2 - polemass * g * polelength * cos_theta
        # E_target = - polemass * g * polelength
        # Etilde = E - E_target
        # Etilde = -g*polelength*polemass + (1/2)*polelength*polemass*(-2*g*cos_theta + polelength*theta_dot*theta_dot)
        # xpp_desired = - K_e * sin_theta * theta_dot * Etilde - K_p * x - K_d * x_dot
        # theta_pp = - cos_theta/polelength * xpp_desired - (g/polelength) * sin_theta
        # f = (cartmass + polemass) * xpp_desired + cos_theta * polemass * polelength* theta_pp - sin_theta * polemass * polelength * theta_dot**2
        # f = K_e * cos_theta * theta_dot * Etilde - K_p * x - K_d * x_dot

    return f, switched


