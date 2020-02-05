import importlib

from uav.controllers.attitude_nmpc import Controller as MS_Controller
importlib.reload(MS_Controller)

import lib.geometry.numpy_geometry as ng
importlib.reload(ng)

from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

def define_interpolation_polynomial(d, poly_type='legendre'):
    """TODO: Docstring for define_interpolation_polynomial.

    :d: Degree of interpolating polynomial
    :poly_type: Type of polynomial
    :returns: TODO

    """

    # Degree of interpolating polynomial
    d = 3

    # Get collocation points
    tau_root = np.append(0, cs.collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    F = dict()
    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

        F[j] = p

    # Plot the polynomials
    if False:
        t = np.arange(0,1,0.01)
        y = dict()
        for j in range(d+1):
            y[j] = F[j](t)
            plt.plot(t, y[j])
        plt.show()

    return {'B':B, 'C':C, 'D':D}

def ocp(data):
    model = data['model']
    Q     = data['Q']
    R     = data['R']

    nx = 10
    nu = 3
    nref = 5

    # States
    x     = cs.SX.sym('x', nx)
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
    u = cs.SX.sym('u', nu)
    delta_a = u[0]
    delta_e = u[1]
    delta_t = u[2]

    # State reference
    ref = cs.SX.sym('ref', nref)
    Vr_ref = ref[0]
    qw_ref = ref[1]
    qx_ref = ref[2]
    qy_ref = ref[3]
    qz_ref = ref[4]

    quat = cs.vertcat(qw, qx, qy, qz)
    s_omega = cs.vertcat(p_s, q_s, r_s)
    input = cs.vertcat(delta_a, delta_e, 0, delta_t)
    # [dot_q_nb, Vr_dot, beta_dot, alpha_dot, dot_s_omega] = dot_x_stability(0, quat, Vr, alpha, beta, s_omega, input, model)
    ode = MS_Controller.dot_x_stability(0, quat, Vr, alpha, beta, s_omega, input, model)


    L = MS_Controller.cost(x, ref, u, data, False, 'full')

    fun = dict()
    fun['cost'] = cs.Function('Cost', [ x, ref, u ], [ L ])
    fun['f']    = cs.Function('f',    [ x, ref, u ], [ ode, L ])

    return (fun['f'], nx, nu, nref)




def OCP(Q, R):
    """TODO: Docstring for OCP.
    :returns: TODO

    """

    # Declare model variables
    x1 = cs.SX.sym('x1')
    x2 = cs.SX.sym('x2')
    x = cs.vertcat(x1, x2)
    u = cs.SX.sym('u')

    ref = cs.SX.sym('ref', x.shape[0])

    # Model equations
    xdot = cs.vertcat((1-x2**2)*x1 - x2 + u, x1)

    # Objective term
    L = cost(x, u, ref, Q, R)

    # Continuous time dynamics
    f = cs.Function('f', [x, u, ref], [xdot, L], ['x', 'u', 'ref'], ['xdot', 'L'])
    return f

def transcribe_idx(w, dim):
    start = cs.vertcat(*w).shape[0]
    return range(start, start + dim)

def transcribe(data, fun, Poly, state_space='full'):

    B = Poly['B']
    C = Poly['C']
    D = Poly['D']

    Q = data['Q']
    R = data['R']

    xlb  = data['x_min']  # [-] State lower bound
    x0   = data['x_init'] # [-] State initial guess
    xub  = data['x_max'] # [-] State upper bound
    nx   = data['nx'] # [-] State dimension

    ulb  = data['u_min']  # [-] Input lower bound
    u0   = data['u_init'] # [-] Input initial guess
    uub  = data['u_max']  # [-] Input upper bound
    nu   = data['nu']     # [-] Input dimension

    nref  = data['nref'] # [-] Reference dimension
    reflb = nref*[0]     # [-] Reference lower bound
    refub = nref*[0]     # [-] Reference upper bound
    ref0  = nref*[0]     # [-] Initial "guess" on reference

    N = data['N'] # [-] Number of control intervalls
    T = data['T'] # [s] Prediction horizon
    d = data['d'] # [-] Order of Lagrange polynomial
    h = T/N       # [s] Length of each control intervall
    f = fun       # [dict] Function dictionary

    # Start with an empty NLP
    w   = [] # Decision variable
    w0  = [] # Decision variable initial guess
    lbw = [] # Decision variable lower bound
    ubw = [] # Decision variable upper bound

    g   = [] # Equality constraints
    lbg = [] # Equality constraints lower bound
    ubg = [] # Equality constriants upper bound

    J = 0 # Cost functional

    # Index of state_k, u_k, ref_k within w
    xidx, uidx, refidx = [], [], []

    # For plotting x and u given w
    x_plot   = []
    u_plot   = []
    ref_plot = []

    # "Lift" initial conditions
    Xk = cs.SX.sym('X0', nx)
    xidx += [range(0,nx)]
    w.append(Xk)
    lbw.append(xlb)
    ubw.append(xub)
    w0.append(x0)
    x_plot.append(Xk)


    # Formulate the NLP
    for k in range(N):

        # New NLP variable for the reference
        refk = cs.SX.sym('ref_' + str(k), nref)
        refidx += [transcribe_idx(w, nref)]
        w.append(refk)
        lbw.append(reflb)
        ubw.append(refub)
        w0.append(ref0)
        ref_plot.append(refk)

        # New NLP variable for the control
        Uk = cs.SX.sym('U_' + str(k), nu)
        uidx += [transcribe_idx(w, nu)]
        w.append(Uk)
        lbw.append(ulb)
        ubw.append(uub)
        w0.append(u0)
        u_plot.append(Uk)

        # State at collocation points
        # (dr): Enfore inequality constraints at collocation points
        Xc = []
        for j in range(d):
            Xkj = cs.SX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append(xlb)
            ubw.append(xub)
            w0.append(x0)

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1,d+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*Xk
           for r in range(d): xp = xp + C[r+1,j]*Xc[r]

           # Append collocation equations
           # Enforce dynamics at collocation points
           fj, qj = f(Xc[j-1], refk, Uk)
           g.append(h*fj - xp)
           lbg.append(nx*[0])
           ubg.append(nx*[0])

           # Add contribution to the end state
           Xk_end = Xk_end + D[j]*Xc[j-1];

           # Add contribution to quadrature function
           J = J + B[j]*qj*h

        # New NLP variable for state at end of interval
        Xk = cs.SX.sym('X_' + str(k+1), nx)
        xidx += [transcribe_idx(w, nx)]
        w.append(Xk)
        lbw.append(xlb)
        ubw.append(xub)
        w0.append(x0)
        x_plot.append(Xk)

        # Add equality constraint (enfore continuity).
        g.append(Xk_end-Xk)
        lbg.append(nx*[0])
        ubg.append(nx*[0])


    refN = cs.SX.sym('ref_' + str(N), nref)
    refidx += [transcribe_idx(w, nref)]
    w.append(refN)
    lbw.append(reflb)
    ubw.append(refub)
    w0.append(ref0)
    ref_plot.append(refN)

    J += MS_Controller.cost(Xk, refN, np.array([0]), data, True, state_space='full')

    # Concatenate vectors
    w   = cs.vertcat(*w)
    w0  = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)

    g   = cs.vertcat(*g)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    x_plot   = cs.horzcat(*x_plot)
    ref_plot = cs.horzcat(*ref_plot)
    u_plot   = cs.horzcat(*u_plot)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    solver = cs.nlpsol('solver', data['solver'], prob, data['solver_opt'])

    nlp_sol = {'solver' : solver,\
           'w0'     : w0,\
           'lbw'    : lbw,\
           'ubw'    : ubw,\
           'lbg'    : lbg,\
           'ubg'    : ubg,\
           'wopt'   : [],\
           'xidx'   : xidx,\
           'refidx' : refidx,\
           'uidx'   : uidx,\
           'nlp'    : prob}


    # Function to get x and u trajectories from w
    trajectories = cs.Function('trajectories', [w], [x_plot, ref_plot, u_plot], ['w'], ['x', 'ref', 'u'])

    return (nlp_sol, trajectories)

class Controller(object):

    """Docstring for Controller. """

    def __init__(self, data):
        """TODO: to be defined1.

        :data: TODO: List fields

        """
        self.data = data
        self.k_ref = range(data['N']+1)

        Poly = define_interpolation_polynomial(data['d'])
        f, nx, nu, nref = ocp(data)
        data['nx']   = nx
        data['nu']   = nu
        data['nref'] = nref

        nlp_sol, self.trajectories = transcribe(data, f, Poly)
        self.nlp_sol = nlp_sol

        self.wOpt   = []
        self.w0     = nlp_sol['w0']
        self.lbw    = nlp_sol['lbw']
        self.ubw    = nlp_sol['ubw']
        self.lbg    = nlp_sol['lbg']
        self.ubg    = nlp_sol['ubg']
        self.xidx   = nlp_sol['xidx']
        self.uidx   = nlp_sol['uidx']
        self.refidx = nlp_sol['refidx']


    def update(self, uav, ref):

        x     = uav.getState()
        alpha = uav.getAOA()
        beta  = uav.getSSA()
        Vr    = uav.getAirspeed()

        nx   = self.data['nx']
        nu   = self.data['nu']
        nref = self.data['nref']

        self.ref = ref[:, self.k_ref]

        # Build initial state x_init as starting point for the NLP.
        quat_nb = x[3:7]
        omega_b = x[10:]
        omega_s = ng.rotation_sb(alpha) @ omega_b;
        x_init = np.concatenate((np.array((Vr, beta, alpha)), quat_nb, omega_s))

        # Set initial conditions
        self.w0[:nx]  = x_init
        self.lbw[:nx] = x_init
        self.ubw[:nx] = x_init

        # Shift previous solution
        if len(self.wOpt) > 0:
            self.w0 = self.wOpt
            x0, ref0, u0 = self.trajectories(self.w0)

        # Fill in reference
        for i, j in enumerate(self.refidx):
            for idx, val in enumerate(j):
                self.w0[val]  = self.ref[idx, i]
                self.lbw[val] = self.ref[idx, i]
                self.ubw[val] = self.ref[idx, i]

        # Check if all bounds are satisfied by initial guess
        # test_ub = self.ubw - self.w0 >= 0
        # test_lb = self.lbw - self.w0 <= 0

        # k = np.arange(0, self.w0.shape[0],1)

        # Sanity check. Plot decision variable and bounds.
        # fig, ax = plt.subplots(3,1, sharex=True)
        # ax[0].plot(k, self.w0)
        # ax[0].plot(k, self.lbw)
        # ax[0].plot(k, self.ubw)
        # ax[1].plot(k, test_lb)
        # ax[2].plot(k, test_ub)
        # ax[0].legend(('w0', 'lbw', 'ubw'))
        # plt.show(block=False)

        sol = self.nlp_sol['solver'](x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        self.wOpt = sol['x'].full().flatten()

        # Plot trajectories
        x_opt, ref_opt, u_opt = self.trajectories(sol['x'])
        x_opt   = x_opt.full()
        u_opt   = u_opt.full()
        ref_opt = ref_opt.full()

        Va_opt      = x_opt[0,:]
        ssa_opt     = x_opt[1,:]
        aoa_opt     = x_opt[2,:]
        quat_opt    = x_opt[3:7,:]
        s_omega_opt = x_opt[7:,:]

        Va_ref = ref_opt[0,:]
        quat_ref = ref_opt[1:,:]

        # Plot the result
        T = self.data['T']
        N = self.data['N']
        tgrid = np.linspace(0, T, N+1)

        if False: # Plot NLP Solution at each iteration
            fig, ax = plt.subplots(3,1)
            ax[0].plot(tgrid, Va_opt)
            ax[0].plot(tgrid, Va_ref, '--')
            ax[1].plot(tgrid, ssa_opt)
            ax[2].plot(tgrid, aoa_opt)

            fig, ax = plt.subplots(4,1)
            for i in range(4):
                ax[i].plot(tgrid, quat_opt[i].T)
                ax[i].plot(tgrid, quat_ref[i].T, '--')

            fig, ax = plt.subplots()
            for i in range(3):
                plt.step(tgrid, np.append(np.nan, u_opt[i,:]), '-.')

        self.u = u_opt[:,0].reshape(u_opt.shape[0])

        # plt.figure(1)
        # plt.clf()
        # plt.plot(tgrid, x_opt[0], '--')
        # plt.plot(tgrid, x_opt[1], '-')
        # plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
        # plt.xlabel('t')
        # plt.legend(['x1','x2','u'])
        # plt.grid()
        # plt.show()

# if False:
#     state_space = 'full'

#     data = dict()
#     data['x_min']  = [-0.25   , -np.inf]
#     data['x_init'] = [0       , 0      ]
#     data['x_max']  = [np.inf  ,  np.inf]
#     data['nx'] = len(data['x_init'])

#     data['u_min']  = [-1]
#     data['u_init'] = [ 0]
#     data['u_max']  = [ 1]
#     data['nu'] = len(data['u_init'])

#     data['nref'] = data['nx']

#     nx   = data['nx']
#     nu   = data['nu']
#     nref = data['nref']

#     data['Q'] = np.identity(nx)
#     data['Q_N'] = np.identity(nx)
#     data['R'] = 0.001*np.identity(nu)
#     d = 3

#     Poly = define_interpolation_polynomial(d)
#     f = ocp(data['Q'], data['R'])

#     T = 20.
#     N = 40 # number of control intervals
#     h = T/N

#     data['T'] = T
#     data['N'] = N
#     ref = np.zeros((2, data['N']+1))

#     ref[1,:] = 0.1*np.ones((1, data['N']+1))
#     opt_nlp_sol = {}

#     controller = Controller(data, opt_nlp_sol)
#     controller.update([0, 0.0], ref)
#     controller.update([0, 0.0], ref)

#     if False:
#         nlp_sol, trajectories = transcribe(data, opt_nlp_sol, f, Poly)

#         solver = nlp_sol['solver']
#         w0 = nlp_sol['w0']
#         lbw = nlp_sol['lbw']
#         ubw = nlp_sol['ubw']
#         lbg = nlp_sol['lbg']
#         ubg = nlp_sol['ubg']

#         # Solve the NLP
#         sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
#         x_opt, u_opt = trajectories(sol['x'])
#         x_opt = x_opt.full() # to numpy array
#         u_opt = u_opt.full() # to numpy array

#         # Plot the result
#         tgrid = np.linspace(0, T, N+1)
#         plt.figure(1)
#         plt.clf()
#         plt.plot(tgrid, x_opt[0], '--')
#         plt.plot(tgrid, x_opt[1], '-')
#         plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
#         plt.xlabel('t')
#         plt.legend(['x1','x2','u'])
#         plt.grid()
#         plt.show()
