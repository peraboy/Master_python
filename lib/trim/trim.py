# The goal is to write a trim routine using casadi

# Module Imports {{{1
import matplotlib.pyplot as plt
import numpy as np

from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs
from lib.geometry import casadi_geometry as cg
from lib.geometry import numpy_geometry as ng
from uav import uav as UAV


def is_zero(var, eps= 1e-12):
    return var < eps

def compute_force(t, quat, Vr, alpha, beta, b_omega, u, model, state_space='full'):

    p = b_omega[0]
    q = b_omega[1]
    r = b_omega[2]
    delta_a = u[0]
    delta_e = u[1]
    delta_r = u[2]
    delta_t = u[3]

    P = model.P

    if state_space == 'full':
        D = model.dragForce(Vr, alpha, beta, q, delta_e, P)
        Y = model.sideForce(Vr, beta, p, r, delta_a, delta_r, P)
        L = model.liftForce(Vr, alpha, q, delta_e, P)
        T = model.thrustForce(Vr, delta_t, P)
        G = P['mass'] * P['gravity']
        l = model.rollMoment(Vr, beta, p, r, delta_a, delta_r, P)
        m = model.pitchMoment(Vr, alpha, q, delta_e, P)
        n = model.yawMoment(Vr, beta, p, r, delta_a, delta_r, P)
    elif state_space == 'longitudinal':
        D = model.dragForce(Vr, alpha, beta, q, delta_e, P)
        Y = 0.0
        L = model.liftForce(Vr, alpha, q, delta_e, P)
        T = model.thrustForce(Vr, delta_t, P)
        l = 0.0
        m = model.pitchMoment(Vr, alpha, q, delta_e, P)
        n = 0.0
    elif state_space == 'lateral':
        D = 0.0
        Y = model.sideForce(Vr, beta, p, r, delta_a, delta_r, P)
        L = 0.0
        T = 0.0
        G = 0.0
        l = model.rollMoment(Vr, beta, p, r, delta_a, delta_r, P)
        m = 0.0
        n = model.yawMoment(Vr, beta, p, r, delta_a, delta_r, P)
    else:
        #TODO: Make this a proper error message.
        print('Unknown state_space')

    G = P['mass'] * P['gravity']

    # Aerodynamic force vector in body-fixed axes components.
    R_wb = cg.rotation_wb(alpha, beta)
    Fa_b = R_wb.T @ cs.vertcat(-D, Y, -L)

    # Propulsion force-vector in body-fixed axes components.
    Ft_b = cs.SX.zeros(3,1)
    Ft_b[0] = T
    # Ft_b = cs.SX((T, 0, 0))

    # Gravity force vector in body-fixed axes components.
    R_nb = cg.rotation_quaternion(quat)
    Fg_n = cs.SX.zeros(3,1)
    Fg_n[2] = G
    Fg_b = R_nb.T @ Fg_n

    # Total force-vector in body-fixed axes c
    # F  = Fa_b.reshape(3) + Ft_b.reshape(3) + Fg_b.reshape(3)
    F  = Fa_b + Ft_b + Fg_b

    # Compute aerodynamic moment-vector in body-fixed axes

    M = cs.vertcat(l, m, n)

    tau = cs.vertcat(F, M)
    return tau



# def force(x, delta, w_n, model):
#     """ Returns a stacked vector [F_b, M_b] of total force & moment vector in R^3 in body-axes. """

#     u = x[3]
#     v = x[4]
#     w = x[5]

#     phi   = x[6]
#     theta = x[7]
#     psi   = x[8]
    
#     p = x[9] 
#     q = x[10]
#     r = x[11]
    
#     delta_a = delta[0]
#     delta_e = delta[1]
#     delta_r = delta[2]
#     delta_t = delta[3]
    
#     vel_r = cs.vertcat(u,v,w) - cg.rotation_rpy(phi, theta, psi).T @ w_n
#     Vr    = cg.airspeed(vel_r)
#     alpha = cg.aoa(vel_r)
#     beta  = cg.ssa(vel_r)
    
#     # Compute force vector
#     D = model.dragForce(Vr, alpha, beta, q, delta_e, model.P)
#     Y = model.sideForce(Vr, beta, p, r, delta_a, delta_r, model.P)
#     L = model.liftForce(Vr, alpha, q, delta_e, model.P)
#     T = model.thrustForce(Vr, delta_t, model.P)
#     G = model.P['mass']*model.P['gravity']

#     R_nb = cg.rotation_rpy(phi, theta, psi)
#     R_wb = cg.rotation_wb(alpha, beta)

#     fA_b = R_wb.T@cs.vertcat(-D, Y, -L)
#     fG_b = R_nb.T@cs.vertcat(0,0,G)
#     fT_b = cs.vertcat(T,0,0)
#     F_b = fA_b + fT_b + fG_b
    
#     # Compute moment vector
#     l = model.rollMoment(Vr, beta, p, r, delta_a, delta_r, model.P)
#     m = model.pitchMoment(Vr, alpha, q, delta_e, model.P)
#     n = model.yawMoment(Vr, beta, p, r, delta_a, delta_r, model.P)
    
#     M_b = cs.vertcat(l,m,n)

#     return cs.vertcat(F_b, M_b)

def T_rpy(roll, pitch, yaw):
    cr = cs.cos(roll)
    sr = cs.sin(roll)
    cp = cs.cos(pitch)
    tp = cs.tan(pitch)
    sp = cs.sin(pitch)
    cy = cs.cos(yaw)
    sy = cs.sin(yaw)
    T  = cs.SX.zeros(3,3)
    T[0,0],T[0,1],T[0,2] = 1, sr*tp, cr*tp
    T[1,0],T[1,1],T[1,2] = 0, cr,    -sr
    T[2,0],T[2,1],T[2,2] = 0, sr/cp, cr/cp
    # T[0,1] = sr*tp 
    # T[0,2] = cr*tp
    # T[1,0] = 0
    # T[1,1] = cr
    # T[1,2] = -sr
    # T[2,0] = 0
    # T[2,1] = sr/cp
    # T[2,2] = cr/cp

    return T

def dynamics(x, tau, P):
    """ Returns the state dynamics based on state x = [p_n, quat_nb, vel_b, angVel_b] and tau = [F_b, M_b]."""
    
    # Fossen eq 3.56
    vel_b = x[3:6]
    Theta = x[6:9]

    angVel_b = x[9:]
    phi   = Theta[0]
    theta = Theta[1]
    psi   = Theta[2]

    r_cg = cs.SX.zeros(3,1)
    c_rb_11 = P['mass']*cg.Smtrx(angVel_b)
    c_rb_12 = cs.SX.zeros(3,3)
    c_rb_21 = cs.SX.zeros(3,3)
    c_rb_22 = -cg.Smtrx(P['I_cg']@angVel_b)

    C_rb = cs.vertcat(cs.horzcat(c_rb_11, c_rb_12),\
                      cs.horzcat(c_rb_21, c_rb_22))


    # Fossen eq. 3.42
    ny = cs.vertcat(vel_b, angVel_b)
    ny_dot = cs.inv(P['M_rb']) @ (tau - C_rb@ny)
    
    vel_b_dot    = ny_dot[0:3]
    angVel_b_dot = ny_dot[3:]
        
    pos_n_dot = cg.rotation_rpy(phi, theta, psi)@vel_b
    Theta_dot = T_rpy(phi, theta, psi)@angVel_b

    xdot = cs.vertcat(pos_n_dot, vel_b_dot, Theta_dot, angVel_b_dot)
    return xdot

def trim(Va_d, R_d, gamma_d, model, print_level=0):
    """Returns trim state and control input for desired airspeed, curve radius and flight-path angle.

    :Va: [m/s] Desired airspeed.
    :R: [rad] Desired curve radius.
    :gamma: [rad] Desired flight-path angle.
    :model: Struct of the UAV model.
    :returns: x_trim = [pos, vel, Euler, ang_vel], u_trim = [delta_a, delta_e, delta_r, delta_t]

    """
    xdot_d = np.array((0, 0, -Va_d*np.sin(gamma_d), 0, 0, 0, 0, 0, Va_d*np.cos(gamma_d)/R_d, 0, 0, 0))

    x = cs.SX.sym('x', 12, 1)
    pN    = x[0]
    pE    = x[1]
    pD    = x[2]
    u     = x[3]
    v     = x[4]
    w     = x[5]
    phi   = x[6]
    theta = x[7]
    psi   = x[8]
    p     = x[9]
    q     = x[10]
    r     = x[11]

    delta = cs.SX.sym('delta', 4, 1)
    delta_a = delta[0]
    delta_e = delta[1]
    delta_r = delta[2]
    delta_t = delta[3]

    n_x = x.shape[0]
    n_u = u.shape[0]

    P = model.Param
    # pN      , pE  , pD   , u    , v   , w   , phi , theta , psi , p   , q   , r   , delta_a , delta_e , delta_r , delta_t
    w0 = [0.0 , 0.0 , -200 , 15 , 0.0 , 0.0 , 0.0 , 0.0   , 0.0 , 0.0 , 0.0 , 0.0 , 0.0     , 0.0     , 0.0     , 0.0]
    lbw = n_x*[-cs.inf] + [P['aileron_min'], P['elevator_min'], P['rudder_min'], P['throttle_min']]
    ubw = n_x*[+cs.inf] + [P['aileron_max'], P['elevator_max'], P['rudder_max'], P['throttle_max']]

    R_nb = cg.rotation_rpy(phi, theta, psi)
    b_v = cs.vertcat(u, v, w)
    n_v = R_nb@b_v
    gamma = cg.flightPathAngle(n_v)
    Vr = cs.sqrt(b_v.T@b_v)

    g = []
    lbg = []
    ubg = []

    g   += [gamma]
    lbg += [gamma_d]
    ubg += [gamma_d]

    g += [Vr]
    lbg += [Va_d]
    ubg += [Va_d]

    Vg = Vr
    chi = psi
    
    # if not R_d == np.inf:
    #     R = Vg**2*cs.cos(gamma)/(model.P['gravity']*cs.tan(phi)*cs.cos(chi - psi))
    #     g += [R]
    #     lbg += [R_d]
    #     ubg += [R_d]

    w_n = cs.SX.zeros(3,1)

    u = x[3]
    v = x[4]
    w = x[5]

    phi   = x[6]
    theta = x[7]
    psi   = x[8]
    
    p = x[9] 
    q = x[10]
    r = x[11]
    
    delta_a = delta[0]
    delta_e = delta[1]
    delta_r = delta[2]
    delta_t = delta[3]
    
    vel_r = cs.vertcat(u,v,w) - cg.rotation_rpy(phi, theta, psi).T @ w_n
    Vr    = cg.airspeed(vel_r)
    alpha = cg.aoa(vel_r)
    beta  = cg.ssa(vel_r)

    b_omega = cs.vertcat(p, q, r)

    tau = compute_force(0, cg.quaternion_rpy(phi, theta, psi), Vr, alpha, beta, b_omega, delta, model, state_space='full')
    dyn = dynamics(x, tau, model.P)

    e = xdot_d[2:] - dyn[2:]
    J = e.T @ e

    w = cs.vertcat(x,delta)
    prob = {'f':J, 'x':w, 'g':cs.vertcat(*g)}
    ipopt_opts = {}
    ipopt_opts = {'print_level':print_level}
    opts = {'ipopt':ipopt_opts}
    solver = cs.nlpsol('solver', 'ipopt', prob, opts);

    # sol = solver(x0=w0, lbx=lbw, ubx=ubw)
    sol = solver(x0=w0, lbg=lbg, ubg=ubg)
    wOpt = sol['x'].full().flatten()

    xOpt = wOpt[0:12]
    uOpt = wOpt[12:]

    v_b = xOpt[3:6]
    Theta_opt = xOpt[6:9]

    phi_opt = Theta_opt[0]
    theta_opt = Theta_opt[1]
    psi_opt = Theta_opt[2]

    v_n = ng.rotation_rpy(phi_opt, theta_opt, psi_opt) @ v_b
    gamma_opt = np.arcsin(-v_n[2]/np.sqrt(v_n.T@v_n))

    omega_b_opt = xOpt[9:]
    u_opt = xOpt[3]
    v_opt = xOpt[4]
    w_opt = xOpt[5]

    # tauOpt = force(xOpt, uOpt, w_n, model)
    # dynOpt = dynamics(xOpt, tauOpt, model.P)
    # print('xdot_d:')
    # print(xdot_d[2:])
    # print('xdot:')
    # print(dynOpt[2:])

    Va_opt = np.sqrt(v_b.T @ v_b)

    x_trim = np.concatenate((wOpt[0:3], ng.quaternion_rpy(phi_opt, theta_opt, psi_opt), v_b, omega_b_opt))
    u_trim = uOpt


    ###################
    v_rel = v_b #- ng.rotation_rpy(phi_opt, theta_opt, psi_opt).T@w_n
    Vr = ng.airspeed(v_rel)
    alpha = ng.aoa(v_rel)
    beta = ng.ssa(v_rel)

    phi = phi_opt
    theta = theta_opt
    psi = psi_opt
    p = omega_b_opt[0]
    q = omega_b_opt[1]
    r = omega_b_opt[2]
    delta_a = u_trim[0]
    delta_e = u_trim[1]
    delta_r = u_trim[2]
    delta_t = u_trim[3]

    pos     = x_trim[0:3]
    quat    = x_trim[3:7] #ng.quaternion_rpy(phi, theta, psi)
    b_v     = x_trim[7:10]
    b_omega = x_trim[10:]

    R_nb = ng.rotation_quaternion(quat)
    n_v = R_nb@b_v

    gamma = ng.flightPathAngle(n_v)

    tau = UAV.compute_force(0, quat, Vr, alpha, beta, b_omega, u_trim, model)

    xdot = UAV.dynamics(0, x_trim, tau, model.P)


    D = model.dragForce(Vr, alpha, beta, q, delta_e, model.P)
    Y = model.sideForce(Vr, beta, p, r, delta_a, delta_r, model.P)
    L = model.liftForce(Vr, alpha, q, delta_e, model.P)
    T = model.thrustForce(Vr, delta_t, model.P)
    G = model.P['mass']*model.P['gravity']

    R_nb = ng.rotation_rpy(phi, theta, psi)
    R_wb = ng.rotation_wb(alpha, beta)

    fA_b = R_wb.T@np.array((-D, Y, -L))
    fG_b = R_nb.T@np.array((0,0,G))
    fT_b = np.array((T,0,0))
    F_b = fA_b + fT_b + fG_b
    
    # Compute moment vector
    l = model.rollMoment(Vr, beta, p, r, delta_a, delta_r, model.P)
    m = model.pitchMoment(Vr, alpha, q, delta_e, model.P)
    n = model.yawMoment(Vr, beta, p, r, delta_a, delta_r, model.P)
    M_b = np.array((l, m, n))

    # Tests
    n_fail = 0
    n_fail += not is_zero(np.linalg.norm(F_b), eps=1e-6)
    n_fail += not is_zero(np.linalg.norm(M_b), eps=1e-6)

    # return (xOpt, uOpt)
    return (x_trim, u_trim)
