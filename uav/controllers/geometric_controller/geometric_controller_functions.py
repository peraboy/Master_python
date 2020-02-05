import numpy as np
from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs
from lib.geometry import casadi_geometry as cg
from lib.geometry import numpy_geometry as ng

def dampingMatrix(Vr, P):
    # Return damping matrix of rotational dynamics.
    D = np.zeros((3,3))
    D[0,0] = P['b']**2*P['C_l_p']
    D[0,2] = P['b']**2*P['C_l_r']
    D[1,1] = P['c']**2*P['C_m_q']
    D[2,0] = P['b']**2*P['C_n_p']
    D[2,2] = P['b']**2*P['C_n_r']
    D *= 0.25*P['rho']*P['S_wing']*Vr

    return D

def controlEffectivenesMatrix(Vr, P):
    # Return control effectivenes matrix of rotational dynamics.
    G = np.zeros((3,3))
    G[0,0] = P['b']*P['C_l_delta_a']
    G[0,2] = P['b']*P['C_l_delta_r']
    G[1,1] = P['c']*P['C_m_delta_e']
    G[2,0] = P['b']*P['C_n_delta_a']
    G[2,2] = P['b']*P['C_n_delta_r']
    G *= 0.5*P['rho']*P['S_wing']*Vr**2
    
    return G

def driftVector(Vr, alpha, beta, P):
    # Return drift vector of rotational dynamics.
    f = np.zeros((3,1))
    f[0,0] = P['b']*(P['C_l_0'] + P['C_l_beta']*beta)
    f[1,0] = P['c']*(P['C_m_0'] + P['C_m_alpha']*alpha)
    f[2,0] = P['b']*(P['C_n_0'] + P['C_n_beta']*beta)
    f *= 0.5*P['rho']*P['S_wing']*Vr**2

    return f

def compute_Psi_E(Gamma, ControlParam):
    return ControlParam['alpha'] + ControlParam['beta']*np.dot(Gamma, ControlParam['sd'])

def compute_Psi_N(Gamma, Gamma_d):
    return 1 - np.dot(Gamma, Gamma_d)

def compute_eSd(Gamma, sd, b, sym_flag = 0):
    if sym_flag:
        eSd = b*cs.cross(sd, Gamma)
    else:
        eSd = b*np.cross(sd, Gamma)

    return eSd


def compute_eH(Gamma, Gamma_d, omega, mode, State, ControlParam):

    Psi_N = compute_Psi_N(Gamma, Gamma_d)
    Psi_E = compute_Psi_E(Gamma, ControlParam)

    sd = ControlParam['sd']
    rho = min(Psi_N, Psi_E)

    hybrid_control = True
    if mode == 0 or hybrid_control == False:
        Psi_m = compute_Psi_N(Gamma, Gamma_d)
    else:
        Psi_m = compute_Psi_E(Gamma, ControlParam)

    if Psi_m - rho >= ControlParam['delta']:
        if mode == 0:
            mode = 1
            Psi_m = Psi_E
            ControlParam['sd'] = compute_optimal_sd(State, ControlParam, option=ControlParam['option'], Q=ControlParam['Q'])
            # if ControlParam['sd_opt'] == 'minimum_control_full':
            #     ControlParam['sd'] = compute_minimum_control_sd(State, ControlParam)
            # elif ControlParam['sd_opt'] == 'geodesic':
            #     ControlParam['sd'] = compute_geodesic_sd(Gamma, Gamma_d)
        else:
            mode = 0
            Psi_m = Psi_N

    if mode == 0 or hybrid_control == False:
        eH = compute_eGamma(Gamma, Gamma_d)
    else:
        eH = compute_eSd(Gamma, ControlParam['sd'], ControlParam['beta'])

    return (mode, eH, ControlParam)

def compute_eGamma(Gamma, Gamma_d):
    return np.cross(Gamma, Gamma_d)

def compute_eOmega(Gamma, omega):
    return -ng.Smtrx(Gamma)@ng.Smtrx(Gamma)@omega

def compute_perpendicular_projection(x, y):
    I = np.eye(len(x))
    y_perp = (I - x@x.T)@y
    return y_perp

def compute_parallel_projection(x, y):
    return np.dot(y, x)*y

def compute_f(omega, Vr, alpha, beta, J, P):
    Jinv = np.linalg.inv(J)
    f = Jinv @ \
            (np.cross(J @ omega,omega) \
            + np.squeeze(driftVector(Vr, alpha, beta, P)) \
            + dampingMatrix(Vr, P)@omega)
    return f

def compute_G(Vr, J, P):
    Jinv = np.linalg.inv(J)
    G = Jinv @ controlEffectivenesMatrix(Vr, P) 
    return G

def compute_control_input(G, f, ControlParam, eH, eOmega, Gamma, omega):
    Ginv = np.linalg.inv(G)
    control = Ginv @ (ng.Smtrx(omega)@eOmega -ControlParam['kGamma']*eH - ControlParam['kOmega']*eOmega + ng.Smtrx(Gamma)@ng.Smtrx(Gamma)@f)
    return control

def compute_geodesic_sd(Gamma, Gamma_d):
    sd = np.cross(np.cross(Gamma_d, Gamma)/np.linalg.norm(np.cross(Gamma_d, Gamma)), Gamma_d)
    return sd


def compute_optimal_sd(State, ControlParam, **kwargs):
    Gamma   = State['Gamma']
    omega   = State['omega']
    Gamma_d = State['Gamma_d']
    Vr      = State['Vr']
    alpha   = State['alpha']
    beta    = State['beta']

    kp = ControlParam['kp']
    kd = ControlParam['kd']
    b  = ControlParam['b']
    I = ControlParam['J']
    P = ControlParam['P']

    sd = cs.SX.sym('sd', 3, 1)

    G      = compute_G(Vr, I, P)
    f      = compute_f(omega, Vr, alpha, beta, I, P)
    eOmega = compute_eOmega(Gamma, omega)
    eSd    = compute_eSd(Gamma, sd, b, sym_flag = 1)


    # Transcribe NLP
    w   = []
    lbw = []
    ubw = []
    g   = []
    lbg = []
    ubg = []
    J = 0

    # Different u, according to option
    if kwargs['option'] == 1:
        # u = -kp*cs.cross(Gamma,sd) - kd*eOmega
        u = -kp*eSd - kd*eOmega
        Q = np.identity(3)
    elif kwargs['option'] == 2:
        # u = -kp*cs.cross(Gamma,sd) - kd*eOmega
        u = -kp*eSd - kd*eOmega
        Q = kwargs['Q']
    elif kwargs['option'] == 3:
        # u = -kp*cs.cross(Gamma,sd) - kd*eOmega
        u = -kp*eSd - kd*eOmega
        Q0 = kwargs['Q']
        Ginv = np.linalg.inv(G)
        Q = I.T @ Ginv.T @ Q0 @ Ginv @ I
        __import__('ipdb').set_trace()
    else:
        u = compute_control_input(G, f, ControlParam, eSd, eOmega, Gamma, omega)
        Q = np.identity(3)


    # Minimum control
    J += 0.5*u.T@Q@u

    # Norm constraint
    g   += [sd.T@sd]
    lbg += [1]
    ubg += [1]

    # Orthogonality constraint
    g   += [sd.T@Gamma_d]
    lbg += [0]
    ubg += [0]

    w += [sd]
    # w0 = ControlParam['sd']
    # w0 = np.array((1,0,0))
    # w0 = compute_geodesic_sd(Gamma, Gamma_d)
    # w0 = Gamma_d
    w0 = Gamma
    # w0 = np.array((1,1,1))
    lbw += [-1, -1, -1]
    ubw += [+1, +1, +1]

    # Create an NLP solver

    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    ipopt_opts = {}
    ipopt_opts = {'print_level':5, 'print_timing_statistics':'no'}
    opts = {'ipopt':ipopt_opts}
    solver = cs.nlpsol('solver', 'ipopt', prob, opts);
    # solver = cs.nlpsol('solver', 'ipopt', prob);

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    sd_opt = np.array(w_opt)
    print('sd_0: ' + str(w0))
    print('sd_opt: ' + str(sd_opt))

    return sd_opt

def compute_minimum_control_sd(State, ControlParam):
    Gamma   = State['Gamma']
    omega   = State['omega']
    Gamma_d = State['Gamma_d']
    Vr      = State['Vr']
    alpha   = State['alpha']
    beta    = State['beta']

    kp = ControlParam['kp']
    kd = ControlParam['kd']
    b  = ControlParam['b']
    J = ControlParam['J']
    P = ControlParam['P']

    sd = cs.SX.sym('sd', 3, 1)

    G      = compute_G(Vr, J, P)
    f      = compute_f(omega, Vr, alpha, beta, J, P)
    eOmega = compute_eOmega(Gamma, omega)
    eSd    = compute_eSd(Gamma, sd, b, sym_flag = 1)
    u      = compute_control_input(G, f, ControlParam, eSd, eOmega, Gamma, omega)

    # Transcribe NLP
    w   = []
    lbw = []
    ubw = []
    g   = []
    lbg = []
    ubg = []
    J = 0

    # Minimum control
    J += 0.5*u.T@u

    # Norm constraint
    g   += [sd.T@sd]
    lbg += [1]
    ubg += [1]

    # Orthogonality constraint
    g   += [sd.T@Gamma_d]
    lbg += [0]
    ubg += [0]

    w += [sd]
    # w0 = ControlParam['sd']
    w0 = np.array((1,0,0))
    lbw += [-1, -1, -1]
    ubw += [+1, +1, +1]

    # Create an NLP solver

    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    ipopt_opts = {}
    ipopt_opts = {'print_level':0, 'print_timing_statistics':'no'}
    opts = {'ipopt':ipopt_opts}
    solver = cs.nlpsol('solver', 'ipopt', prob, opts);
    # solver = cs.nlpsol('solver', 'ipopt', prob);

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    sd_opt = np.array(w_opt)
    # print('sd_opt: ' + str(sd_opt))

    return sd_opt
