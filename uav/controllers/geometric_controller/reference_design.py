import numpy as np
import matplotlib.pyplot as plt
import importlib
import os as os
from uav.controllers.geometric_controller import controller as Geometric_controller
importlib.reload(Geometric_controller)


from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs

from lib.geometry import casadi_geometry as cg
importlib.reload(cg)


def is_orthogonal(x, y):
    """TODO: Docstring for is_orthogonal.

    :x: TODO
    :y: TODO
    :returns: TODO

    """

    return np.dot(x, y) < 1e-16
def test_eval(val, eps=1e-16):
    """TODO: Docstring for test_eval.

    :val: TODO
    :returns: TODO

    """
    if val < eps:
        success = True
    else:
        success = False

    fail = not success

    return fail

def design_reference_analytic(phi_0, A_phi, f_phi, theta_0, A_theta, f_theta, time, run_test=True):


    def compute_Gamma(phi, theta):

        # sp = np.sin(phi)
        # cp = np.cos(phi)
        # st = np.sin(theta)
        # ct = np.cos(theta)

        N = len(phi)
        Gamma = np.zeros((3, N))
        Gamma[0, :] = -np.sin(theta)
        Gamma[1, :] = np.cos(theta)*np.sin(phi)
        Gamma[2, :] = np.cos(theta)*np.cos(phi)

        return Gamma

    def compute_Gamma_dot(phi, phi_dot, theta, theta_dot):

        sp = np.sin(phi)
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)

        N = len(phi)
        Gamma_dot = np.zeros((3, N))
        Gamma_dot[0, :] = -ct*theta
        Gamma_dot[1, :] = -st*sp*theta_dot + ct*cp*phi_dot
        Gamma_dot[2, :] = -st*cp*theta_dot - ct*sp*phi_dot
        return Gamma_dot

    def compute_Gamma_ddot(phi, phi_dot, phi_ddot, theta, theta_dot, theta_ddot):

        sp = np.sin(phi)
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)

        N = len(phi)
        Gamma_ddot = np.zeros((3, N))
        Gamma_ddot[0, :] = -st*theta_dot**2 -ct*theta_ddot
        Gamma_ddot[1, :] = -ct*sp*(theta_dot**2 + phi_dot**2) - 2*st*cp*theta_dot*phi_dot \
                - st*sp*theta_ddot + ct*cp*phi_ddot
        Gamma_ddot[2, :] = -ct*cp*(theta_dot**2 + phi_dot**2) + 2*st*sp*theta_dot*phi_dot \
                - st*cp*theta_ddot - ct*sp*phi_ddot

        return Gamma_ddot


    A = np.array((A_phi, A_theta))
    f = np.array((f_phi, f_theta))
    Omega = 2*np.pi*f
    # Omega_phi = 2*np.pi*f_phi
    # Omega_theta = 2*np.pi*f_theta

    t = time
    theta      = theta_0 + A[0]             * np.sin(Omega[0]*t)
    theta_dot  =       + A[0]*Omega[0]    * np.cos(Omega[0]*t)
    theta_ddot =       - A[0]*Omega[0]**2 * np.sin(Omega[0]*t)

    phi      = phi_0 + A[1]             * np.cos(Omega[1]*t)
    phi_dot  =         - A[1]*Omega[1]    * np.sin(Omega[1]*t)
    phi_ddot =         - A[1]*Omega[1]**2 * np.cos(Omega[1]*t)

    Gamma      = compute_Gamma(phi, theta)
    Gamma_dot  = compute_Gamma_dot(phi, phi_dot, theta, theta_dot)
    Gamma_ddot = compute_Gamma_ddot(phi, phi_dot, phi_ddot, theta, theta_dot, theta_ddot)

    N = len(t)
    omega = np.zeros((3, N))
    omega_dot = np.zeros((3, N))
    for i in range(0, N):
        omega[:, i]     = np.cross(Gamma_dot[:, i], Gamma[:, i])
        omega_dot[:, i] = np.cross(Gamma_ddot[:, i], Gamma[:, i])

    if run_test:

        is_orthogonal_Gamma_Gamma_dot = 0
        is_orthogonal_Gamma_omega = 0
        N = len(t)
        for i in range(0, N):
            is_orthogonal_Gamma_Gamma_dot += np.dot(Gamma[:, i], Gamma_dot[:, i])**2
            is_orthogonal_Gamma_omega     += np.dot(Gamma[:, i], omega[:, i])**2

        print('Failed tests:' + str(test_eval(is_orthogonal_Gamma_Gamma_dot) +test_eval(is_orthogonal_Gamma_omega)))

    return [Gamma, omega, omega_dot]






def design_reference(phi_0, A_phi, f_phi, theta_0, A_theta, f_theta, time, run_test=False):
    """TODO: Docstring for design_reference.

    :phi_0: TODO
    :A_phi: TODO
    :f_phi: TODO
    :theta_0: TODO
    :A_theta: TODO
    :f_theta: TODO
    :returns: TODO

    """

    def evaluate_scalar_casadi_function(fun, var):
        """TODO: Docstring for evaluate_scalar_casadi_function.

        :fun: TODO
        :var: TODO
        :returns: TODO

        """
        return np.squeeze(np.double(fun(var)))

    def f_eval_cs_vec(fun, var):
        """TODO: Docstring for f_eval_cs_vec.

        :fun: TODO
        :var: TODO
        :returns: TODO

        """
        return np.double(fun(var).reshape((3,1))).reshape(3)
    
    # phi_0 = 0
    # A_phi = np.pi/6
    # f_phi = 0.5

    # theta_0 = 0
    # A_theta = np.pi/8
    # f_theta = 0.5

    # Symbolics
    t = cs.SX.sym('t',1,1)

    phi = phi_0 + A_phi * cs.sin(2*cs.pi*f_phi*t)
    dphi  = cs.jacobian(phi, t)
    ddphi = cs.jacobian(dphi, t)

    theta = theta_0 + A_theta * cs.cos(2*cs.pi*f_theta*t)
    dtheta  = cs.jacobian(theta, t)
    ddtheta = cs.jacobian(dtheta, t)

    Gamma     = Geometric_controller.compute_sym_Gamma_rp(phi, theta)
    # Gamma_dot = cg.orthogonal_projector(Gamma)@cs.jacobian(Gamma, t)
    Gamma_dot = cs.jacobian(Gamma, t)
    omega     = cs.cross(Gamma_dot, Gamma)
    omega_dot = cs.jacobian(omega, t)

    f_phi       = cs.Function('phi'       , [t] , [phi])
    f_theta     = cs.Function('theta'     , [t] , [theta])
    f_Gamma     = cs.Function('Gamma'     , [t] , [Gamma])
    f_Gamma_dot = cs.Function('Gamma_dot' , [t] , [Gamma_dot])
    f_omega     = cs.Function('omega'     , [t] , [omega])
    f_omega_dot = cs.Function('omega_dot' , [t] , [omega_dot])

    # Gamma_dot = cg.compute_perpendicular_projector(Gamma)@

    # Generate numeric values
    t = time
    t0 = t[0]
    dt = t[1] - t[0]
    fs = 1/dt
    T = t[-1]  + dt
    N = len(t)

    phi = np.zeros(N)
    theta = np.zeros(N)
    Gamma = np.zeros((3, N))
    Gamma_dot = np.zeros((3, N))
    omega = np.zeros((3, N))
    omega_dot = np.zeros((3, N))

    phi          = np.squeeze(np.double(f_phi(t)))
    theta        = np.squeeze(np.double(f_theta(t)))

    is_orthogonal_Gamma_omega = np.zeros(N)

    test_orthogonal_Gamma_omega = 0
    test_orthogonal_Gamma_Gamma_dot = 0
    for k in range(0, N):
        Gamma[:, k]     = f_eval_cs_vec(f_Gamma,     t[k])
        Gamma_dot[:, k] = f_eval_cs_vec(f_Gamma_dot, t[k])
        omega[:, k]     = f_eval_cs_vec(f_omega,     t[k])
        omega_dot[:, k] = f_eval_cs_vec(f_omega_dot, t[k])

        test_orthogonal_Gamma_omega += np.dot(Gamma[:, k], omega[:, k])**2
        test_orthogonal_Gamma_Gamma_dot += np.dot(Gamma[:, k], Gamma_dot[:, k])**2


    if run_test:
        print('Design reference')
        print('# Tests failed:' + str(test_eval(test_orthogonal_Gamma_omega) + test_eval(test_orthogonal_Gamma_Gamma_dot)))

        # Integration Test
        Gamma_integrated = np.zeros((3, N))
        omega_integrated = np.zeros((3, N))
        x_Gamma = Gamma[:, 0] #Geometric_controller.compute_Gamma_rp(phi[0], theta[0])
        x_omega = omega[:, 0]

        for k in range(0, N):
            Gamma_dot = np.cross(Gamma[:, k], omega[:, k])
            x_Gamma += dt * Gamma_dot
            x_Gamma = x_Gamma/np.linalg.norm(x_Gamma)
            Gamma_integrated[:, k] = x_Gamma

            x_omega += dt * omega_dot[:, k]
            # x_omega += Geometric_controller.compute_perpendicular_projector(Gamma[:, k])@x_omega
            omega_integrated[:, k] = x_omega

        fig, ax = plt.subplots(2, 1)
        color = ['r', 'g', 'b']
        for i in range(0,3):
            ax[0].plot(t, Gamma[i, :].T, color=color[i])
            ax[0].plot(t, Gamma_integrated[i, :].T, color=color[i], linestyle='--')
            ax[1].plot(t, omega[i, :].T, color=color[i])
            ax[1].plot(t, omega_integrated[i, :].T, color=color[i], linestyle='--')
            plt.show(block=False)


    return [Gamma, omega, omega_dot]


    # phi     = evaluate_scalar_casadi_function(fun_phi['f'], t)
    # dphi    = evaluate_scalar_casadi_function(fun_phi['df'], t)
    # ddphi   = evaluate_scalar_casadi_function(fun_phi['ddf'], t)

    # theta   = evaluate_scalar_casadi_function(fun_theta['f'], t)
    # dtheta  = evaluate_scalar_casadi_function(fun_theta['df'], t)
    # ddtheta = evaluate_scalar_casadi_function(fun_theta['ddf'], t)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(t, phi)
    # ax[0].plot(t, dphi)
    # ax[0].plot(t, ddphi)
    # ax[1].plot(t, theta)
    # ax[1].plot(t, dtheta)
    # ax[1].plot(t, ddtheta)
    # # plt.show()

    # Gamma = np.zeros((3, N))
    # dGamma = np.zeros((3, N))
    # dGamma_proj = np.zeros((3, N))
    # ddGamma = np.zeros((3, N))
    # omega = np.zeros((3, N))
    # domega = np.zeros((3, N))


    # for k in range(0, N):
    #     Gamma[:, k]  = Geometric_controller.compute_Gamma_rp(phi[k], theta[k])
    #     dGamma[:, k] = Geometric_controller.compute_Gamma_rp_dot(phi[k], dphi[k], theta[k], dtheta[k])
    #     # omega[:, k]  = Geometric_controller.compute_omega_rp(Gamma[:, k], dGamma[:, k])
    #     omega[:, k] = Geometric_controller.compute_omega_rp(phi[k], dphi[k], dtheta[k])
    #     # omega[:, k] = Geometric_controller.compute_omega_rp_proj(phi[k], dphi[k], theta[k], dtheta[k])
    #     domega[:, k] = Geometric_controller.compute_omega_rp_dot_perp(phi[k], dphi[k], ddphi[k], theta[k], dtheta[k], ddtheta[k])

    #     # print(is_orthogonal(Gamma[:, k], dGamma[:, k]))


    #     # ddGamma[:, k] = Geometric_controller.compute_Gamma_rp_ddot(phi[k], dphi[k], ddphi[k], theta[k], dtheta[k], ddtheta[k])
    #     # domega[:, k] = Geometric_controller.compute_omega_rp_dot(Gamma[:, k], ddGamma[:, k])

    # # fig, ax = plt.subplots()
    # # ax.plot(t, dGamma.T)
    # # ax.plot(t, dGamma_proj.T)
    # # plt.show()

    # # Integrate test
    # test = {'Gamma':np.zeros((3, N)),\
    #         'dGamma':np.zeros((3, N)),\
    #         'ddGamma':np.zeros((3, N)),\
    #         'omega':np.zeros((3, N)),\
    #         'domega':np.zeros((3, N))}

    # for k in range(0, N):
    #     test['dGamma'][:, k] = np.cross(Gamma[:, k], omega[:, k])
    #     # test['ddGama'][:, k] = np.cross(test['dGamma'],omega
        
    # fig, ax = plt.subplots()
    # ax.plot(t, dGamma.T)
    # ax.plot(t, test['dGamma'].T, linestyle='--')
    # plt.show()








