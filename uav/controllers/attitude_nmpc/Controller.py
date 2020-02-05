import numpy as np
import importlib
import casadi as cs
from lib.geometry import casadi_geometry as cg
importlib.reload(cg)
from lib.geometry import numpy_geometry as ng
importlib.reload(ng)

import uav.uav as UAV


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

def s_angvel_sb(alpha_dot):
    return b_angvel_sb(alpha_dot)

def b_angvel_sb(alpha_dot):
    # Stevens, Lewis (2.5-12a)
    b_angvel_sb = cs.SX.zeros(3,1)
    b_angvel_sb[1] = -alpha_dot
    return b_angvel_sb

def s_inertia_cg(alpha, b_J):
    R_sb = cg.rotation_sb(alpha)
    s_J = R_sb @ b_J @ R_sb.T
    return s_J

def dot_s_angvel_bn(alpha, alpha_dot, b_M, b_omega_bn, b_J_cg) :
    R_sb = cg.rotation_sb(alpha);
    s_J_cg = s_inertia_cg(alpha, b_J_cg)
    s_M = R_sb @ b_M;

    s_omega_bn = R_sb @ b_omega_bn
    s_omega_sb = s_angvel_sb(alpha_dot)

    dot_s_omega_bn = - cg.Smtrx(s_omega_sb)@s_omega_bn + cs.inv(s_J_cg)@(s_M - cg.Smtrx(s_omega_bn)@s_J_cg@s_omega_bn)

    return dot_s_omega_bn

def dot_x_stability(t,quat, Vr, alpha, beta, s_omega, u, model):
    P = model.P

    R_sb = cg.rotation_sb(alpha)
    R_ws = cg.rotation_ws(beta)
    R_wb = cg.rotation_wb(alpha, beta)

    b_omega = R_sb.T @ s_omega
    p = b_omega[0]
    q = b_omega[1]
    r = b_omega[2]
    delta_a = u[0]
    delta_e = u[1]
    delta_r = u[2]
    delta_t = u[3]

    b_tau = compute_force(0, quat, Vr, alpha, beta, b_omega, u, model)
    b_F = b_tau[:3]
    b_M = b_tau[3:]

    w_F     = R_wb @ b_F
    w_omega = R_wb @ b_omega
    w_v_rel = cs.SX.zeros(3,1)
    w_v_rel[0] = Vr
    # w_v_rel = np.array((Vr,0,0)) # [Vr, 0, 0].T
    # rhs     = w_F/P['mass'] - cg.Smtrx(w_omega)@w_v_rel
    rhs     = w_F/P['mass'] - cs.cross(w_omega, w_v_rel)

    Vr_dot    = rhs[0]
    beta_dot  = rhs[1]/Vr
    alpha_dot = rhs[2]/(Vr*cs.cos(beta))

    b_J_cg = P['I_cg']

    dot_s_omega = dot_s_angvel_bn(alpha, alpha_dot, b_M, b_omega, b_J_cg)

    dot_quat_nb = cg.quaternion_dot(quat, b_omega)
        # dot_q_nb = dot_q_nb(:);
    s_x_dot = cs.vertcat(Vr_dot, beta_dot, alpha_dot, dot_quat_nb, dot_s_omega)


    return s_x_dot
    # return [dot_q_nb, Vr_dot, beta_dot, alpha_dot, dot_s_omega]
    # return [dot_q_nb, Vr_dot, beta_dot, alpha_dot, dot_s_omega]


def cost(x, x_ref, u, OCP, is_terminal, state_space):
    x_tilde = compute_x_tilde(x, x_ref, state_space);

    if not is_terminal:
        J = x_tilde.T@OCP['Q']@x_tilde + u.T@OCP['R']@u;
    else:
        J = x_tilde.T@OCP['Q_N']@x_tilde;

    return J

def numpy_compute_x_tilde(x, x_ref, state_space):

    if state_space == 'longitudinal':
        Vr    = x[0];
        alpha = x[1];
        theta = x[2];
        q     = x[3];

        Vr_ref    = x_ref[0];
        theta_ref = x_ref[1];

        x_tilde = cs.vertcat(Vr - Vr_ref, alpha, theta - theta_ref, q)

    elif state_space == 'full':
        Vr_err = x[0] - x_ref[0]

        q = x[3:7]
        q_ref = x_ref[1:5]

        # Compute reduced attitudes
        R_d = ng.rotation_quaternion(q_ref)
        R   = ng.rotation_quaternion(q)

        b = R_d[:, 0]

        Gamma_d = R_d.T@b;
        Gamma   = R.T@b;

        # Cost functions
        q_err = np.zeros(4)
        q_err[0] = 1 - np.dot(np.array((1, 0, 0)) , Gamma)
        # q_err = 1 - Gamma[0]
        # q_err = cs.vertcat(q_err, cs.cross(Gamma_d, Gamma))
        q_err[1:] = np.cross(Gamma_d, Gamma)

        # q_err = np.vstack((q_err, cs.cross(Gamma_d, Gamma))


        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        phi = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx**2 + qy**2))
        omega_s = x[7:10]

        beta = x[1]
        # x_tilde = np.vstack((Vr_err, q_err, omega_s, phi, beta))
        x_tilde = np.zeros(10)
        x_tilde[0] = Vr_err
        x_tilde[1:5] = q_err
        x_tilde[5:8] = omega_s
        x_tilde[8] = phi
        x_tilde[9] = beta
    elif state_space == 'lateral':
        print('Not implemented state_space: ' + state_space)
    else:
        print('Unknown state_space: ' + state_space)

    return x_tilde





def compute_x_tilde(x, x_ref, state_space):

    if state_space == 'longitudinal':
        Vr    = x[0];
        alpha = x[1];
        theta = x[2];
        q     = x[3];

        Vr_ref    = x_ref[0];
        theta_ref = x_ref[1];

        x_tilde = cs.vertcat(Vr - Vr_ref, alpha, theta - theta_ref, q)

    elif state_space == 'full':
        Vr_err = x[0] - x_ref[0]

        q = x[3:7]
        q_ref = x_ref[1:5]

        # Compute reduced attitudes
        R_d = cg.rotation_quaternion(q_ref)
        R   = cg.rotation_quaternion(q)

        b = R_d[:, 0]

        Gamma_d = R_d.T@b;
        Gamma   = R.T@b;

        # Cost functions
        q_err = 1 - cs.dot(cs.vertcat(1, 0, 0) , Gamma)
        # q_err = 1 - Gamma[0]
        q_err = cs.vertcat(q_err, cs.cross(Gamma_d, Gamma))


        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        phi = cs.arctan2(2*(qw*qx + qy*qz), 1-2*(qx**2 + qy**2))
        omega_s = x[7:10]

        beta = x[1]
        x_tilde = cs.vertcat(Vr_err, q_err, omega_s, phi, beta)
    elif state_space == 'lateral':
        print('Not implemented state_space: ' + state_space)
    else:
        print('Unknown state_space: ' + state_space)

    return x_tilde


def dot_airspeed(Vr, alpha, beta, qw, qx, qy, qz, q, delta_e, delta_t, model, use_small_angle=False):
    P = model.P
    D = model.dragForce(Vr, alpha, beta, q, delta_e, P)
    T = model.thrustForce(Vr, delta_t, P)
    g_w = compute_g_wind(alpha, beta, cs.vertcat(qw, qx, qy, qz), P['gravity'])
    gx_w = g_w[0]

    if use_small_angle:
        sa = alpha;
        ca = 1;
        sb = beta;
        cb = 1;
    else:
        sa = cs.sin(alpha);
        ca = cs.cos(alpha);
        sb = cs.sin(beta);
        cb = cs.cos(beta);

    P = model.P
    dot_Vr = (1/P['mass']) * (T * ca * cb - D + P['mass'] * gx_w);
    return dot_Vr

def compute_g_wind(alpha, beta, quat, g):
    R_nb = cg.rotation_quaternion(quat)
    R_wb = cg.rotation_wb(alpha, beta)
    R_wn = R_wb@R_nb.T
    g_w = R_wn@cs.vertcat(0,0,g)
    return g_w

def compute_gx_wind(alpha, beta, qw, qx, qy, qz, P):
    pass
def compute_gy_wind(alpha, beta, qw, qx, qy, qz, P):
    pass
def compute_gz_wind(alpha, beta, qw, qx, qy, qz, P):
    pass

def dot_aoa(Vr, alpha, beta, qw, qx, qy, qz, p_s, q, delta_e, delta_t, model, use_small_angle):
    P = model.P
    L = model.liftForce(Vr, alpha, q, delta_e, P)
    T = model.thrustForce(Vr, delta_t, P)
    g_w = compute_g_wind(alpha, beta, cs.vertcat(qw, qx, qy, qz), P['gravity'])
    gz_w = g_w[2]

    if use_small_angle:
        sa = alpha;
        ca = 1;
        sb = beta;
        cb = 1;
    else:
        sa = cs.sin(alpha);
        ca = cs.cos(alpha);
        sb = cs.sin(beta);
        cb = cs.cos(beta);

    dot_alpha = \
        (1/(P['mass'] * (Vr) * cb)) * (-T *sa - L + \
        P['mass'] * gz_w) + (q * cb - p_s * sb) / cb;

    return dot_alpha

def dot_s_pitchrate(Vr, alpha, p_s, q, r_s, delta_e, model, use_small_angle):
    P = model.P
    m = model.pitchMoment(Vr, alpha, q, delta_e, P);

    s_J = s_inertia_cg(alpha, P['I_cg'])
    s_J_xx = s_J[0,0]
    s_J_yy = s_J[1,1]
    s_J_zz = s_J[2,2]
    s_J_xz = s_J[0,2]

    dot_s_q = \
         (m - p_s * (s_J_xx * r_s + s_J_xz * p_s) + \
         r_s * (s_J_xz * r_s + s_J_zz * p_s)) / s_J_yy
    return dot_s_q

def ocp(data, OCP):
    model = data['model']
    Q     = OCP['Q']
    R     = OCP['R']

    # States
    x     = cs.SX.sym('x', 10)
    Vr    = x[0]
    beta  = x[1]
    alpha = x[2]
    qw    = x[3]
    qx    = x[4]
    qy    = x[5]
    qz    = x[6]
    p_s   = x[7]
    q_s   = x[8]
    r_s   = x[9]

    # Control inputs
    u = cs.SX.sym('u', 3)
    delta_a = u[0]
    delta_e = u[1]
    delta_t = u[2]

    # State reference
    x_ref = cs.SX.sym('xRef', 5)
    Vr_ref = x_ref[0]
    qw_ref = x_ref[1]
    qx_ref = x_ref[2]
    qy_ref = x_ref[3]
    qz_ref = x_ref[4]

    quat = cs.vertcat(qw, qx, qy, qz)
    s_omega = cs.vertcat(p_s, q_s, r_s)
    input = cs.vertcat(delta_a, delta_e, 0, delta_t)
    # [dot_q_nb, Vr_dot, beta_dot, alpha_dot, dot_s_omega] = dot_x_stability(0, quat, Vr, alpha, beta, s_omega, input, model)
    ode = dot_x_stability(0, quat, Vr, alpha, beta, s_omega, input, model)
    # ode = cs.vertcat(Vr_dot, alpha_dot, beta_dot, dot_q_nb, dot_s_omega)

    OCP['x'] = x
    OCP['u'] = u
    OCP['xRef'] = x_ref

    L = cost(x, x_ref, u, OCP, False, 'full')

    fun = dict()

    fun['cost'] = cs.Function('Cost', [ x, x_ref, u ], [ L ])
    fun['f']    = cs.Function('f',    [ x, x_ref, u ], [ ode, L ])
    OCP['ode'] = ode
    OCP['L'] = L

    return (OCP, fun)

def ocp_longitudinal(data, OCP):
    model = data['model']
    Q = OCP['Q']
    R = OCP['R']

    # State
    x     = cs.SX.sym('x', 4)
    Vr    = x[0]
    alpha = x[1]
    theta = x[2]
    q     = x[3]

    # Input
    u = cs.SX.sym('u', 2)
    delta_e = u[0]
    delta_t = u[1]

    # Reference
    x_ref     = cs.SX.sym('x_ref', 2)
    Vr_ref    = x_ref[0]
    theta_ref = x_ref[1]

    # Dynamic model
    quat = cg.quaternion_rpy(0, theta, 0)
    qw   = quat[0]
    qx   = quat[1]
    qy   = quat[2]
    qz   = quat[3]

    Euler = cg.rpy_quaternion(quat)

    beta = 0
    r_s  = 0
    q_s  = q
    p_s  = 0

    delta = cs.SX.zeros(4)
    delta[1] = delta_e
    delta[3] = delta_t
    s_omega = cs.SX.zeros(3)
    s_omega[1] = q
    # use_small_angle = false

    dot_x_s = dot_x_stability(0, quat, Vr, alpha, beta, s_omega, delta, model)

    Euler = cg.rpy_quaternion(quat)
    use_small_angle=False
    P = model.P
    ode = cs.vertcat(\
        dot_airspeed(Vr, alpha, beta, qw, qx, qy, qz, q_s, delta_e, delta_t, model, use_small_angle),\
        dot_aoa(Vr, alpha, beta, qw, qx, qy, qz, p_s, q_s, delta_e, delta_t, model, use_small_angle),\
        q,\
        dot_s_pitchrate(Vr, alpha, p_s, q_s, r_s, delta_e, model, use_small_angle))
    ode_old = cs.vertcat(dot_x_s[0], dot_x_s[1], Euler[1], dot_x_s[-2])

    OCP['x'] = x
    OCP['u'] = u
    OCP['xRef'] = x_ref

    L = cost(x, x_ref, u, OCP, False, 'longitudinal')

    fun = dict()

    fun['cost'] = cs.Function('Cost', [ x, x_ref, u ], [ L ])
    fun['f'] = cs.Function('f', [ x, x_ref, u ], [ ode, L ])
    OCP['ode'] = ode
    OCP['L'] = L

    return OCP, fun

def ocp_longitudinal(data, OCP):
    model = data['model']
    Q = OCP['Q']
    R = OCP['R']

    # State
    x     = cs.SX.sym('x', 4)
    Vr    = x[0]
    alpha = x[1]
    theta = x[2]
    q     = x[3]

    # Input
    u = cs.SX.sym('u', 2)
    delta_e = u[0]
    delta_t = u[1]

    # Reference
    x_ref     = cs.SX.sym('x_ref', 2)
    Vr_ref    = x_ref[0]
    theta_ref = x_ref[1]

    # Dynamic model
    quat = cg.quaternion_rpy(0, theta, 0)
    qw   = quat[0]
    qx   = quat[1]
    qy   = quat[2]
    qz   = quat[3]

    Euler = cg.rpy_quaternion(quat)

    beta = 0
    r_s  = 0
    q_s  = q
    p_s  = 0

    delta = cs.SX.zeros(4)
    delta[1] = delta_e
    delta[3] = delta_t
    s_omega = cs.SX.zeros(3)
    s_omega[1] = q
    # use_small_angle = false

    dot_x_s = dot_x_stability(0, quat, Vr, alpha, beta, s_omega, delta, model)

    Euler = cg.rpy_quaternion(quat)
    use_small_angle=False
    P = model.P
    ode = cs.vertcat(\
        dot_airspeed(Vr, alpha, beta, qw, qx, qy, qz, q_s, delta_e, delta_t, model, use_small_angle),\
        dot_aoa(Vr, alpha, beta, qw, qx, qy, qz, p_s, q_s, delta_e, delta_t, model, use_small_angle),\
        q,\
        dot_s_pitchrate(Vr, alpha, p_s, q_s, r_s, delta_e, model, use_small_angle))
    ode_old = cs.vertcat(dot_x_s[0], dot_x_s[1], Euler[1], dot_x_s[-2])

    OCP['x'] = x
    OCP['u'] = u
    OCP['xRef'] = x_ref

    L = cost(x, x_ref, u, OCP, False, 'longitudinal')

    fun = dict()

    fun['cost'] = cs.Function('Cost', [ x, x_ref, u ], [ L ])
    fun['f'] = cs.Function('f', [ x, x_ref, u ], [ ode, L ])
    OCP['ode'] = ode
    OCP['L'] = L

    return OCP, fun
    

def erk4(f, T, N, n_x, n_u, n_ref):
    #ERK4 Implements a 4th order explicit Runge-Kutta integrator.
    #   Intgrates state ode and associated cost.

    # Fixed step Runge-Kutta 4 integrator
    M = 4; # RK4 steps per interval
    DT = T/N/M;
    X0 = cs.SX.sym('X0', n_x);
    U  = cs.SX.sym('U',  n_u);
    X = X0;
    X_ref = cs.SX.sym('X_ref', n_ref);
    Q = 0;
    for j in range(M):
        k1, k1_q = f(X, X_ref, U);
        k2, k2_q = f(X + DT/2 * k1, X_ref, U);
        k3, k3_q = f(X + DT/2 * k2, X_ref, U);
        k4, k4_q = f(X + DT * k3, X_ref, U);
        X = X + DT/6*(k1   + 2*k2   + 2*k3   + k4);
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);

    F = cs.Function('F', [ X0, X_ref, U ], [ X, Q ], [ 'x0', 'ref', 'p' ], [ 'xf', 'qf' ]);

    return F

def cost_du(du, Q_dU):
    return du.T@Q_dU@du

def transcribe_idx(w, dim):
    start = cs.vertcat(*w).shape[0]
    return range(start, start + dim)

def transcribe(data, opt_nlp_sol, OCP, fun, nX, nU, nXref, state_space):

    # OCP Weighting matrices
    R    = OCP['R']
    Q_dU = OCP['Q_dU']
    Q_N  = OCP['Q_N']
    W    = OCP['W']

    # Constraint backoff parameter
    eps  = OCP['eps']

    # Start with an empty NLP
    w   = []
    w0  = []
    lbw = []
    ubw = []
    J   = 0
    g   = []
    lbg = []
    ubg = []


    # "Lift" initial conditions.
    Xk = cs.MX.sym('X0', nX)

    # Fix first NLP variable to initial state
    w   += [Xk]
    w0  += data['x_init']
    lbw += data['x_init']
    ubw += data['x_init']

    ## Slack variables for relaxed inequality constraints.
    nS = 4
    S = cs.MX.sym('S', nS)

    refIdx = []#cell(self.OCP.N+1, 1);
    uIdx   = []
    xIdx   = []

    w_u_idx = []

    # for k=0:self.OCP.N-1
    for k in range(OCP['N']):

        Xrefk = cs.MX.sym('Xref_' + str(k), nXref);

        refIdx += [transcribe_idx(w, nXref)]
        w   += [Xrefk]
        lbw += nXref*[0]
        ubw += nXref*[0]
        w0  += nXref*[0]

        # New NLP variable for the control
        Uk = cs.MX.sym('U_' + str(k), nU)
        uIdx += [transcribe_idx(w, nU)]
        w += [Uk]
        w_u_idx += [len(w)-1]

        lbw += data['u_min']
        ubw += data['u_max']
        w0  += data['u_init']

        # Integrate till the end of the interval
        Fk = fun['F'](x0=Xk, ref=Xrefk, p=Uk)
        Xk_end = Fk['xf']
        #L = cost(x, ref, u, Q, R);
        #Jk = self.OCP.L(Xk, Xrefk, Uk, Q, R);
        Jk = fun['cost'](Xk, Xrefk, Uk)
        J = J + Jk

        # New NLP variable for state at end of interval
        Xk = cs.MX.sym('X_' + str(k+1), nX)
        xIdx += [transcribe_idx(w, nX)]
        w += [Xk]
        lbw += data['x_min']
        ubw += data['x_max']
        w0  += data['x_init']

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += nX*[0]
        ubg += nX*[0]


    # Penalize deviations in U
    uStep = w_u_idx[1] - w_u_idx[0];
    for iU in w_u_idx[1:]:
        J = J + cost_du(w[iU] - w[iU - uStep], Q_dU)

    # Add terminal cost
    xN = Xk;
    XrefN = cs.MX.sym('Xref_' + str(OCP['N']), nXref);

    refIdx += [transcribe_idx(w, nXref)]
    w += [XrefN]
    lbw += nXref*[0]
    ubw += nXref*[0]
    w0  += nXref*[0]

    J_N = cost(xN, XrefN, 0.0, OCP, True, state_space)
    J  = J + J_N

    w_nlp = cs.vertcat(*w)
    g_nlp = cs.vertcat(*g)
    J_nlp = J;


    #TODO: Implement Sanity check
    # g_nlp_1 = vertcat(g{1});
    # w_nlp_1 = vertcat(w{1});

    # # Sanity check
    # jac = csi.Function('jac_g_x', {w_nlp_1}, {jacobian(g_nlp_1, w_nlp)});

    # self.jac_g_x  = csi.Function('jac_g_x'  , {w_nlp}, {jacobian(g_nlp , w_nlp)});
    # self.grad_f_x = csi.Function('grad_f_x' , {w_nlp}, {jacobian(J_nlp , w_nlp)});



    nlp = {'f':J, 'x':w_nlp, 'g':g_nlp}

    solver = cs.nlpsol('solver', 'ipopt', nlp, opt_nlp_sol)

    nlp_sol = {'solver' : solver,\
               'w0'     : w0,\
               'lbw'    : lbw,\
               'ubw'    : ubw,\
               'lbg'    : lbg,\
               'ubg'    : ubg,\
               'wOpt'   : [],\
               'xIdx'   : xIdx,\
               'refIdx' : refIdx,\
               'uIdx'   : uIdx,\
               'nlp'    : nlp}

    return nlp_sol


def xMPC(x, Vr, alpha, beta, state_space):
    # xMPC transforms the vehicle state ([pos, quat, linVel_b, angVel_b]) into
    # the state of the mpc model ([Vr, aoa, ssa, quat, angVel_s])

    quat_nb = x[3:7]
    omega_b = x[10:]
    omega_s = ng.rotation_sb(alpha) @ omega_b;
    if state_space == 'longitudinal':
        Euler = ng.rpy_quaternion(quat_nb)
        q = omega_b[1]
        xMPC = np.array((Vr, alpha, Euler[1], q))
    else:
        xMPC = np.concatenate((np.array((Vr, alpha, beta)), quat_nb, omega_s))
    return xMPC

class Controller(object):

    """Docstring for Controller. """

    def __init__(self, data, opt, opt_nlp_sol, state_space='full', fs_sim=100):
        """TODO: to be defined1. """
        self.data = data
        self.opt  = opt

        self.OCP = opt['OCP']


        if state_space == 'longitudinal':
            res = ocp_longitudinal(self.data, self.OCP)
        else:
            res = ocp(self.data, self.OCP)

        self.OCP = res[0]
        self.fun = res[1]

        self.nX    = self.OCP['x'].shape[0];
        self.nU    = self.OCP['u'].shape[0];
        self.nXref = self.OCP['xRef'].shape[0];
        self.state_space = state_space;

        #self.erk4();
        #self.n_x   = numel(self.OCP.x);
        #self.n_u   = numel(self.OCP.u);
        #self.n_ref = numel(self.OCP.xRef);

        self.fun['F'] = erk4(self.fun['f'], self.OCP['T'], self.OCP['N'], self.nX, self.nU, self.nXref)

        #TODO: Migrate this test from MATLAB
        # test_result = test_ode_and_integrator(self.fun.f, self.fun.F, data.Param);

        self.nlp_sol = transcribe(self.data, opt_nlp_sol, self.OCP, self.fun, self.nX, self.nU, self.nXref, state_space)
        self.refIdx = self.nlp_sol['refIdx']
        self.xIdx   = self.nlp_sol['xIdx']
        self.uIdx   = self.nlp_sol['uIdx']
        self.lbw    = self.nlp_sol['lbw']
        self.ubw    = self.nlp_sol['ubw']
        self.lbg    = self.nlp_sol['lbg']
        self.ubg    = self.nlp_sol['ubg']
        self.nlp    = self.nlp_sol['nlp']
        self.solver = self.nlp_sol['solver']
        self.wOpt = []
        self.w0     = self.nlp_sol['w0']

        self.fs = opt['fs']
        self.dec_fac = fs_sim/self.fs

        self.k_ref = np.arange(0,self.OCP['N']+1) * self.dec_fac
        self.k_ref = self.k_ref.astype(int)

        self.dt = self.OCP['T']/self.OCP['N']

    def update(self, uav, ref):

        x     = uav.getState()
        alpha = uav.getAOA()
        beta  = uav.getSSA()
        Vr    = uav.getAirspeed()


        if len(self.wOpt) > 0:
            # Shift previous solution
            # self.w0 = self.wOpt#[self.nX+self.nXref+self.nU:] + self
            n_u = self.nU;
            n_x = self.nX;
            n_ref = self.nXref;

            self.w0 = np.concatenate((self.wOpt[n_x+n_u+n_ref:] , \
                    self.wOpt[self.uIdx[-1]] , \
                    self.wOpt[self.xIdx[-1]] , \
                    self.wOpt[self.refIdx[-1]])).tolist()

        self.ref = ref[:, self.k_ref]
        
        
        # Build initial state x_init as starting point for the NLP.
        # x_init = xMPC(x, Vr, alpha, beta, self.state_space)
        quat_nb = x[3:7]
        omega_b = x[10:]
        omega_s = ng.rotation_sb(alpha) @ omega_b;
        xMPC = np.concatenate((np.array((Vr, beta, alpha)), quat_nb, omega_s))

        x_init = xMPC

        # Initial conditions
        self.w0[:self.nX]  = x_init
        self.lbw[:self.nX] = x_init
        self.ubw[:self.nX] = x_init

        for i, j in enumerate(self.refIdx):
            for idx, val in enumerate(j):
                self.w0[val]  = self.ref[idx, i]
                self.lbw[val] = self.ref[idx, i]
                self.ubw[val] = self.ref[idx, i]

        # The reference enters the NLP through inequality constraints. Thus
        # at the according indices we need to set the initial value of the
        # decision variable and the lower and upper bound to the reference.
        # The same with the current state.
        # if self.state_space == 'longitudinal':
            # Set reference

            if False: #len(self.nlp_sol['wOpt']) > 0:
                #self.shift()

                # TODO: (dr) Export index shifting to a seperate function. Also include state and input shift!
                # TODO: (dr) Close to solution, but lacks shifting.
                self.w0 = self.wOpt;

                self.nlp_sol['w0'][0:self.nX] = x_init

                self.nlp_sol['w0'][self.nX+self.nXref+1:self.nX+self.nXref+self.nU] = np.zeros(1,2); #[self.u(2), self.u(4)];


        # else:            
            # pass
            #TODO: Implement this. But check if transcribe can be written general enough such that cases are not needed anymore.
                #if(~isempty(self.wOpt))
                #    #self.shift()

                #    # TODO: (dr) Export index shifting to a seperate function. Also include state and input shift!
                #    # TODO: (dr) Close to solution, but lacks shifting.
                #    self.nlp_sol['w0'] = self.wOpt;

                #    self.nlp_sol['w0'](1:self.nX) = self.x_init;
                #    self.nlp_sol['w0'](self.nX+self.nXref+1:self.nX+self.nXref+self.nU) = [self.u(1), self.u(2), self.u(4)];
                #    self.lbw(1:self.nX) = self.x_init;
                #    self.ubw(1:self.nX) = self.x_init;
                #    for i = 1:self.OCP.N+1
                #        from = self.refIdx{i}(1);
                #        to = self.refIdx{i}(end);
                #        self.nlp_sol['w0'](from:to)  = ref(i, :);
                #        self.lbw(from:to) = ref(i, :);
                #        self.ubw(from:to) = ref(i, :);
                #    end
                #end
        
        # self.w0 = self.nlp_sol['w0']
        self.sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw,lbg=self.lbg, ubg=self.ubg)
        
        ## Extract variables from vector of decision variables
        self.wOpt = self.sol['x'].full().flatten()
        #self.u = [self.wOpt(16), self.wOpt(17), 0, self.wOpt(18)];
        fro = self.nX+self.nXref
        if self.state_space == 'full':
            to = fro+3
        elif self.state_space == 'longitudinal':
            to = fro+2
        self.u = self.wOpt[fro:to].T
        return self.u
