import numpy as np
from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs
from lib.geometry import casadi_geometry as cg
from lib.geometry import numpy_geometry as ng

import uav.uav as UAV

def compute_Gamma_rp(roll_angle, pitch_angle):
    Gamma = np.zeros(3)
    Gamma[0] = -np.sin(pitch_angle)
    Gamma[1] = np.cos(pitch_angle)*np.sin(roll_angle)
    Gamma[2] = np.cos(pitch_angle)*np.cos(roll_angle)

    return Gamma

def compute_sym_Gamma_rp(roll_angle, pitch_angle):
    Gamma = cs.SX.zeros(3)
    Gamma[0] = -cs.sin(pitch_angle)
    Gamma[1] = cs.cos(pitch_angle)*cs.sin(roll_angle)
    Gamma[2] = cs.cos(pitch_angle)*cs.cos(roll_angle)

    # Gamma = cs.vertcat(-cs.sin(theta), cs.cos(theta)*cs.sin(phi), cs.cos(theta)*cs.cos(phi))
    return Gamma

def compute_sym_Gamma_rp_dot(roll_angle, roll_angle_dot, pitch_angle, pitch_angle_dot):
    """TODO: Docstring for compute_Gamma_rp.

    :roll_angle: TODO
    :pitch_angle: TODO
    :roll_angle_dot: TODO
    :pitch_angle_dot: TODO
    :returns: TODO

    """
    sr = cs.sin(roll_angle)
    cr = cs.cos(roll_angle)
    sp = cs.sin(pitch_angle)
    cp = cs.cos(pitch_angle)
    p_dot = pitch_angle_dot
    r_dot = roll_angle_dot

    Gamma_dot = cs.zeros(3)
    Gamma_dot[0] = -cp*r_dot
    Gamma_dot[1] = -sp*sr*p_dot + cp*cr*r_dot
    Gamma_dot[2] = -sp*cr*p_dot - cp*sr*r_dot

    Gamma = compute_sym_Gamma_rp(roll_angle, pitch_angle)
    Pi_perp = cs.perpendicular_projector(Gamma)
    Gamma_dot_proj = Pi_perp @ Gamma_dot

    return Gamma_dot_proj

def compute_Gamma_rp_dot(roll_angle, roll_angle_dot, pitch_angle, pitch_angle_dot):
    """TODO: Docstring for compute_Gamma_rp.

    :roll_angle: TODO
    :pitch_angle: TODO
    :roll_angle_dot: TODO
    :pitch_angle_dot: TODO
    :returns: TODO

    """
    sr = np.sin(roll_angle)
    cr = np.cos(roll_angle)
    sp = np.sin(pitch_angle)
    cp = np.cos(pitch_angle)
    p_dot = pitch_angle_dot
    r_dot = roll_angle_dot

    Gamma_dot = np.zeros(3)
    Gamma_dot[0] = -cp*r_dot
    Gamma_dot[1] = -sp*sr*p_dot + cp*cr*r_dot
    Gamma_dot[2] = -sp*cr*p_dot - cp*sr*r_dot

    Gamma = compute_Gamma_rp(roll_angle, pitch_angle)
    Pi_perp = compute_perpendicular_projector(Gamma)
    Gamma_dot_proj = Pi_perp @ Gamma_dot

    return Gamma_dot_proj

def compute_Gamma_rp_ddot(roll_angle, roll_angle_dot, roll_angle_ddot, pitch_angle, pitch_angle_dot, pitch_angle_ddot):
    """TODO: Docstring for compute_Gamma_rp_ddot.

    :roll_angle: TODO
    :roll_angle_dot: TODO
    :roll_angle_ddot: TODO
    :pitch_angle: TODO
    :pitch_angle_dot: TODO
    :pitch_angle_ddot: TODO
    :returns: TODO

    """
    sr = np.sin(roll_angle)
    cr = np.cos(roll_angle)
    sp = np.sin(pitch_angle)
    cp = np.cos(pitch_angle)
    dp = pitch_angle_dot
    dr = roll_angle_dot
    ddp = pitch_angle_ddot
    ddr = roll_angle_ddot

    Gamma_ddot = np.zeros(3)
    Gamma_ddot[0] = sp*dp**2 - cp*ddp
    Gamma_ddot[1] = -cp*sr*(dp**2 + dr**2) - 2*sp*cr*dr*dp 
    Gamma_ddot[2] = -cp*cr*(dp**2 + dr**2) + 2*sp*sr*dr*dp

    return Gamma_ddot

def compute_omega_rp(roll_angle, roll_angle_dot, pitch_angle_dot):
    """TODO: Docstring for compute_omega_rp.

    :roll_angle: TODO
    :roll_angle_dot: TODO
    :pitch_angle_dot: TODO
    :returns: TODO

    """
    sr = np.sin(roll_angle)
    cr = np.cos(roll_angle)
    dp = pitch_angle_dot
    dr = roll_angle_dot

    omega = np.zeros(3)
    omega[0] = dr
    omega[1] = cr*dp
    omega[2] = -sr*dp

    return omega

def compute_omega_rp_dot(roll_angle, roll_angle_dot, roll_angle_ddot, pitch_angle, pitch_angle_dot, pitch_angle_ddot):
    """TODO: Docstring for compute_omega_rp_dot.

    :roll_angle: TODO
    :roll_angle_dot: TODO
    :roll_angle_ddot: TODO
    :pitch_angle: TODO
    :pitch_angle_dot: TODO
    :pitch_angle_ddot: TODO
    :returns: TODO

    """
    sr = np.sin(roll_angle)
    cr = np.cos(roll_angle)
    sp = np.sin(pitch_angle)
    cp = np.cos(pitch_angle)
    dp = pitch_angle_dot
    dr = roll_angle_dot
    ddp = pitch_angle_ddot
    ddr = roll_angle_ddot

    omega_dot = np.zeros(3)
    omega_dot[0] = ddp
    omega_dot[1] = cr*ddp - sr*dr*dp
    omega_dot[2] = -sr*ddp - cr*dr*dp

    return omega_dot

def compute_omega_rp_dot_perp(roll_angle, roll_angle_dot, roll_angle_ddot, pitch_angle, pitch_angle_dot, pitch_angle_ddot):
    """TODO: Docstring for compute_omega_rp_dot_perp.

    :roll_angle: TODO
    :roll_angle_dot: TODO
    :roll_angle_ddot: TODO
    :pitch_angle: TODO
    :pitch_angle_dot: TODO
    :pitch_angle_ddot: TODO
    :returns: TODO

    """
    sr = np.sin(roll_angle)
    cr = np.cos(roll_angle)
    sp = np.sin(pitch_angle)
    cp = np.cos(pitch_angle)
    dp = pitch_angle_dot
    dr = roll_angle_dot
    ddp = pitch_angle_ddot
    ddr = roll_angle_ddot


    Gamma = compute_Gamma_rp(roll_angle, pitch_angle)
    Gamma_dot = compute_Gamma_rp_dot(roll_angle, roll_angle_dot, pitch_angle, pitch_angle_dot)
    omega = compute_omega_rp(roll_angle, roll_angle_dot, pitch_angle_dot)
    omega_dot = compute_omega_rp_dot(roll_angle, roll_angle_dot, roll_angle_ddot, pitch_angle, pitch_angle_dot, pitch_angle_ddot)
    Pi_perp = compute_perpendicular_projector(Gamma)

    omega_dot_perp = Pi_perp@omega_dot - np.dot(omega, Gamma)*Gamma_dot
    return omega_dot_perp

def compute_omega_rp_proj(roll_angle, roll_angle_dot, pitch_angle, pitch_angle_dot):
    """TODO: Docstring for compute_omega_rp.

    :roll_angle: TODO
    :roll_angle_dot: TODO
    :pitch_angle_dot: TODO
    :returns: TODO

    """
    Gamma = compute_Gamma_rp(roll_angle, pitch_angle)
    omega = compute_omega_rp(roll_angle, roll_angle_dot, pitch_angle_dot)
    Pi_perp = compute_perpendicular_projector(Gamma)
    omega_proj = Pi_perp @ omega

    return omega_proj

def compute_omega_rp_old(Gamma_rp, Gamma_rp_dot):
    """TODO: Docstring for compute_omega_rp_dot.

    :Gamma_rp: TODO
    :Gamma_rp_dot: TODO
    :returns: TODO

    """
    return np.cross(Gamma_rp_dot, Gamma_rp)

def compute_omega_rp_dot_old(Gamma_rp, Gamma_rp_ddot):
    """TODO: Docstring for compute_omega_rp_dot.

    :Gamma_rp: TODO
    :Gamma_rp_ddot: TODO
    :returns: TODO

    """
    return np.cross(Gamma_rp_ddot, Gamma_rp)

def compute_synergy_gap(mode, F):
    return F[mode] - min(F)

def compute_sigma(mode):
    return mode

def compute_potential_smoothed(U, mode, p, Lambda):
    sigma = compute_sigma(mode)
    W = U + 0.5*(p - sigma)*Lambda*(p - sigma)
    return W

def compute_potential_basic_gradient(self, mode):
    pass

def compute_error_vector(mode, Gamma, ref, b):
    Gamma_d = ref[0]
    s_d = ref[1]
    if mode == 0:
        e = compute_eGamma(Gamma, Gamma_d)
    elif mode == 1:
        e = compute_eSd(Gamma, s_d, b)
    else:
        raise Exception('Mode ' + str(mode) + ' not implemented.')

    return e

def compute_nominal_potential(Gamma, Gamma_d):
    return 1 - np.dot(Gamma, Gamma_d)

def compute_potential_basic(mode, Gamma, ref, Param):
    if mode == 0:
        potential = compute_nominal_potential(Gamma, ref[0])
        # potential = 1 - np.dot(Gamma, ref[0])
    elif mode == 1:
        potential = Param['a'] + Param['b']*(1- np.dot(Gamma, ref[1]))
    else:
        raise Exception('Mode ' + str(mode) + ' not implemented.')

    return potential

def compute_mode(mode, gap, e_omega, Param):
    potential_condition = gap > Param['delta']
    velocity_condition = np.linalg.norm(e_omega) <= Param['B_e_omega']#True
    if potential_condition and velocity_condition and Param['hybrid_control']:
        mode = 1 - mode
    return mode

def compute_geodesic_versor(Gamma, Gamma_d):
    return np.cross(np.cross(Gamma_d, Gamma)/np.linalg.norm(np.cross(Gamma_d, Gamma)), Gamma_d)

def compute_eOmega(Gamma, omega, omega_d):
    Pi_perp = compute_perpendicular_projector(Gamma)
    # return Pi_perp@omega - Pi_perp@omega_d
    return -ng.Smtrx(Gamma)@ng.Smtrx(Gamma)@(omega - omega_d)

def compute_eGamma(Gamma, Gamma_d):
    return np.cross(Gamma, Gamma_d)

def compute_eSd(Gamma, sd, b, sym_flag = 0):
    if sym_flag:
        eSd = b*cs.cross(Gamma, sd)
    else:
        eSd = b*np.cross(Gamma, sd)

    return eSd

def compute_perpendicular_projector(x):
    # Assumes x has norm equal to one
    # return np.eye(len(x)) - x@x.T
    return -ng.Smtrx(x)@ng.Smtrx(x)

def compute_parallel_projector(x):
    # Assumes x has norm equal to one
    return np.outer(x, x)

def compute_error_vector_smoothed(p, e):
    # e[0]: nominal error vector
    # e[1]: expelling error vector
    # p: Dynamic logic variable
    return (1 - p)*e[0] + p*e[1]


def compute_p_dot(p, sigma, e_Gamma, e_omega, Param):
    Lambda = Param['Lambda']
    Phi    = Param['Phi']
    theta = e_Gamma[0] - e_Gamma[1] 
    p_dot = -(1/Lambda)*np.dot(theta, e_omega) - Phi*(p - sigma)
    return p_dot

def update_p(p, p_dot, fs):
    p += (1/fs)*p_dot
    return p

def compute_Lyapunov(U, e_omega, e_Gamma, ControlParam):
    # U:       Potential in current mode
    # e_omega: Angular velocity error
    # e_Gamma: Error vector in current mode
    V = 0.5*np.dot(e_omega, e_omega) + ControlParam['c']*np.dot(e_omega, e_Gamma) + ControlParam['kp']*U
    return V

def compute_f(omega, drift_force, damping_matrix, Jinv):
    return Jinv@(drift_force + damping_matrix@omega)

def compute_G(controlEffectivenesMatrix, Jinv):
    return Jinv@(controlEffectivenesMatrix)

def compute_angle(Gamma, Gamma_d):
    """Computes the angle between the current reduced attitude and the reference.

    :Gamma: TODO
    :Gamma_d: TODO
    :returns: TODO

    """
    return np.arccos(np.dot(Gamma, Gamma_d))

def compute_f_and_G(omega, driftVector, dampingMatrix, controlEffectivenesMatrix, Jinv):
    f = Jinv @ (driftVector + dampingMatrix@omega)
    G = Jinv @ controlEffectivenesMatrix     
    return (f, G)

def compute_kappa(Gamma, omega, omega_d, omega_d_dot, e_Gamma, e_omega, Param):
    Pi_perp = compute_perpendicular_projector(Gamma)
    Pi_par  = compute_parallel_projector(Gamma)
    omega_perp = Pi_perp@omega
    omega_par = Pi_par@omega
    kp = Param['kp']
    Kd = Param['Kd']

    kappa = -np.cross(omega_perp,omega_par - Pi_par@omega_d) \
            + Pi_perp@omega_d_dot \
            - kp*e_Gamma \
            - Pi_perp@Kd@e_omega
    return kappa

def compute_control_input(G, f, Param, e_Gamma, e_omega, Gamma, omega, omega_d, omega_d_dot):
    Ginv = np.linalg.inv(G)
    J = Param['J']
    Jinv = Param['Jinv']
    Pi_perp = compute_perpendicular_projector(Gamma)
    omega_perp = Pi_perp@omega

    kappa = compute_kappa(Gamma, omega, omega_d, omega_d_dot, e_Gamma, e_omega, Param)
    control = Ginv @ J @ (-Pi_perp@Jinv@(f + np.cross(J @ omega,omega)) + kappa)
    return control

def update_sd(sd, omega_d):
    self.sd += np.cross(sd, omega_d)/self.ControlParam['fs']

class Controller(object):
    def __init__(self, uav, Gamma_d_init, ControlParam):

        R = uav.getRotation_nb()
        Gamma = R.T[:,2]
        # Reference direction of the repelling mode
        self.mode = 0 # Start in nominal mode
        self.p = compute_sigma(self.mode)

        self.ControlParam = ControlParam
        self.ControlParam['Jinv'] = np.linalg.inv(self.ControlParam['J'])

        self.State = dict()
        self.sd = compute_geodesic_versor(Gamma, Gamma_d_init)
        self.e = 2*[np.zeros(3)]
        self.U = 2*[0]
        self.W = 2*[0]
        self.synergy_gap_basic = 0
        self.synergy_gap_smoothed = 0
        self.check = True

    def update(self, uav, Gamma_d, omega_d, omega_d_dot):
        R = uav.getRotation_nb()
        Gamma = R.T[:,2]

        P = self.ControlParam['P']

        omega = np.squeeze(uav.getAngularVelocity())
        Vr    = uav.getAirspeed()
        aoa = uav.getAOA()
        ssa  = uav.getSSA()
        damping_matrix = UAV.dampingMatrix(Vr, P)
        G = UAV.controlEffectivenesMatrix(Vr, P)
        drift_force = np.squeeze(UAV.driftVector(Vr, aoa, ssa, P))


        f = compute_f(omega, drift_force, damping_matrix, self.ControlParam['Jinv'])

        self.Theta = compute_angle(Gamma, Gamma_d)
        eOmega = compute_eOmega(Gamma, omega, omega_d)
        self.eOmega = eOmega

        ref = [Gamma_d, self.sd]
        for k in range(0,2):
            self.U[k] = compute_potential_basic(k, Gamma, ref, self.ControlParam)
            self.W[k] = compute_potential_smoothed(self.U[k], self.mode, self.p, self.ControlParam['Lambda'])
            self.e[k] = compute_error_vector(k, Gamma, ref, self.ControlParam['b'])

        self.synergy_gap_basic    = compute_synergy_gap(self.mode, self.U)
        self.synergy_gap_smoothed = compute_synergy_gap(self.mode, self.W)

        # Switch?
        if self.ControlParam['hybrid_smoothed']:
            gap = self.synergy_gap_smoothed
        else:
            gap = self.synergy_gap_basic

        new_mode = compute_mode(self.mode, gap, eOmega, self.ControlParam)

        ControlParam = self.ControlParam

        if new_mode - self.mode == 1:
            if ControlParam['option'] == 1:
                self.sd = compute_geodesic_versor(Gamma, Gamma_d)
            else:
                self.State['Gamma']   = Gamma   
                self.State['omega']   = omega   
                self.State['Gamma_d'] = Gamma_d 
                self.State['Vr']      = Vr      
                self.State['aoa']   = aoa   
                self.State['ssa']    = ssa    
                self.sd = self.compute_optimal_sd(self.State, ControlParam, option=ControlParam['option'], Q=ControlParam['Q'])

        mode = new_mode
        p_dot = compute_p_dot(self.p, compute_sigma(mode), self.e, self.eOmega, self.ControlParam)
        self.p = update_p(self.p, p_dot, self.ControlParam['fs'])

        e_smoothed = compute_error_vector_smoothed(self.p, self.e)

        pot = self.U[mode]
        e_Gamma = self.e[mode]

        if ControlParam['hybrid_control'] and ControlParam['hybrid_smoothed']:
            if self.check:
                print('using smooth')
                self.check = False
            pot = self.W[mode]
            e_Gamma = e_smoothed

        self.e_smoothed = e_smoothed

        self.e_Gamma = e_Gamma
        self.V = compute_Lyapunov(pot, self.eOmega, e_Gamma, self.ControlParam)
        self.control = compute_control_input(G, f, self.ControlParam, e_Gamma, eOmega, Gamma, omega, omega_d, omega_d_dot)

        self.mode = mode

        u = self.control
        J = self.ControlParam['J']
        Jinv = self.ControlParam['Jinv']

        J = self.ControlParam['J']
        omega_dot_model = self.ControlParam['Jinv']@np.cross(J@omega,omega) + f + G@self.control.reshape(3)

    def compute_optimal_sd(self, State, ControlParam, **kwargs):
        Gamma   = State['Gamma']
        omega   = State['omega']
        Gamma_d = State['Gamma_d']
        Vr      = State['Vr']
        aoa   = State['aoa']
        ssa    = State['ssa']

        kp = ControlParam['kp']
        kd = ControlParam['kd']
        b  = ControlParam['b']
        I  = ControlParam['J']
        P  = ControlParam['P']

        sd = cs.SX.sym('sd', 3, 1)
        omega_d = cs.SX.zeros(3, 1)

        G      = self.compute_G(Vr, I, P)
        f      = self.compute_f(omega, Vr, aoa, ssa, I, P)
        eOmega = self.compute_eOmega(Gamma, omega, omega_d)
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
        if kwargs['option'] == 2:
            # u = -kp*cs.cross(Gamma,sd) - kd*eOmega
            u = -kp*eSd - kd*eOmega
            Q = np.identity(3)
        elif kwargs['option'] == 3:
            # u = -kp*cs.cross(Gamma,sd) - kd*eOmega
            u = -kp*eSd - kd*eOmega
            Q = kwargs['Q']
        elif kwargs['option'] == 4:
            # u = -kp*cs.cross(Gamma,sd) - kd*eOmega
            u = -kp*eSd - kd*eOmega
            Q0 = kwargs['Q']
            Ginv = np.linalg.inv(G)
            Q = I.T @ Ginv.T @ Q0 @ Ginv @ I
        elif kwargs['option'] == 5:
            u = self.compute_control_input(G, f, ControlParam, eSd, eOmega, Gamma, omega)
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
        w0 = Gamma_d
        # w0 = Gamma
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

    def compute_minimum_control_sd(self, State, ControlParam):
        Gamma   = State['Gamma']
        omega   = State['omega']
        Gamma_d = State['Gamma_d']
        Vr      = State['Vr']
        aoa   = State['aoa']
        ssa    = State['ssa']

        kp = ControlParam['kp']
        kd = ControlParam['kd']
        b  = ControlParam['b']
        J = ControlParam['J']
        P = ControlParam['P']

        sd = cs.SX.sym('sd', 3, 1)

        G      = compute_G(Vr, J, P)
        f      = compute_f(omega, Vr, aoa, ssa, J, P)
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
