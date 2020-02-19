import numpy as np
# SIMULATION parameter box______________________________________________________________________________________________
tFinal   = 30.0 # Seconds simulated
T = 0.05
# Pitch ref step
p_step = np.array([13, 3, 13, -3, 20]) * (np.pi / 180)

# Wishlist plot 1
wear = False
actuator_dynamic = True
pl1 = [wear, actuator_dynamic]

# Wishlist plot 2
wear = True
actuator_dynamic = True
pl2 = [wear, actuator_dynamic]
# _______________________________________________________________________________________________________________________

total = dict()
for zz in range(2):
    if zz == 0:
        wear = pl1[0]
        actuator_dynamic = pl1[1]
    else:
        wear = pl1[0]
        actuator_dynamic = pl1[1]
    import numpy as np
    import scipy.io as sio
    import time
    import matplotlib.pyplot as plt
    from uav.controllers import PID
    from uav.models import parameters
    import uav.models.X8 as model
    import uav.sensors.accelerometer as accelerometer
    from uav import uav as uav
    import evaluation.plot
    import importlib

    from lib.geometry import numpy_geometry as ng
    from lib.trim import trim

    ## CASADI PATH:
    from sys import path
    path.append(r"home/perasta/anaconda3/lib/python3.7/site-packages/casadi")

    # Import turbulence from MATLAB simulink, (Dryden gust model)
    from scipy.io import loadmat
    windgust = loadmat('WindGust_moderate.mat')
    windgust = windgust['windGust_moderate']


    importlib.reload(trim)
    importlib.reload(uav)

    #Simulation setup
    RadDeg = 180/np.pi
    tInit = 0.0

    dt = 0.01
    fsSim = int(1/dt)
    t = np.arange(tInit, tFinal, dt)
    nSample = t.size
    N = nSample
    wind_n = np.array([-5.0, 2.0, 0.5])

    if wear:
        model.P['wear'] = True
    else:
        model.P['wear'] = False

    model.Param = model.P #parameters.loadParameters('uav/models/gryteX8_param.mat')
    model.Param['rudder_min'] = model.Param['rudder_max'] = 0

    # Trim
    Va = 18
    R = np.inf
    gamma = 0
    res = trim.trim(Va, R, gamma, model,print_level=5)

    x_trim = res[0]
    u_trim = res[1]
    x_init = x_trim
    u_init = u_trim


    # Controller
    P = model.P # Dictionary
    pidRoll  = PID(0.78,   0.01, -0.11, -0.09, 0.09, 0.1, P['aileron_min'], P['aileron_max'])
    pidPitch = PID(-0.78, -0.30, -0.16, -0.1, 0.1, 0.1, P['elevator_min'], P['elevator_max'])
    pidYaw   = PID(1.08,   0.36,   0.0, -0.09, 0.09, 0.1)
    pidV     = PID(0.69,   1.0,   0.0, -0.9, 0.9, 1.0, P['throttle_min'], P['throttle_max'])

    X        = np.zeros([13,nSample])
    U        = np.zeros([4,nSample])
    F        = np.zeros([3,nSample])
    M        = np.zeros([3,nSample])
    airspeed = np.zeros(nSample)
    aoa      = np.zeros(nSample)
    ssa      = np.zeros(nSample)
    v_r      = np.zeros(nSample)

    state_space = 'full'

    # Reference ________________________________________________________________________________________________________
    ref_roll = np.zeros(nSample)
    ref_pitch = np.zeros(nSample)
    # ref_pitch[round(nSample/2):] = (20*np.pi/180)*np.ones(round(nSample/2))
    ref_yaw = np.zeros(nSample)
    ref_V  = Va*np.ones(nSample)

    # Varying references:-----------------------------------------------------------------------------------------------
    counter = 0
    for step in p_step:
        start = counter
        end = counter + 600
        ref_pitch[start:end] = step
        counter += 600




    # Add more to include prediction horizon _______________________________________________________________________________
    ref_roll  = np.concatenate((ref_roll  , ref_roll[-1]*np.ones(round(nSample/2))))
    ref_pitch = np.concatenate((ref_pitch , ref_pitch[-1]*np.ones(round(nSample/2))))
    ref_yaw   = np.concatenate((ref_yaw   , ref_yaw[-1]*np.ones(round(nSample/2))))
    ref_V     = np.concatenate((ref_V     , ref_V[-1]*np.ones(round(nSample/2))))

    # TODO: Apply low-pass filter to reference
    # TODO: Shift solution in controller to use previous w as w0
    if state_space == 'full':
        ref_quat = np.zeros((4, len(ref_roll)))
        for i in range(0, len(ref_roll)):
            ref_quat[:,i] = ng.quaternion_rpy(ref_roll[i], ref_pitch[i], ref_yaw[i])
        ref = np.vstack((ref_V.reshape(1,len(ref_V)), ref_quat))

    elif state_space == 'longitudinal':

        ref = np.vstack((ref_V.reshape(1,len(ref_V)), ref_pitch.reshape(1,len(ref_pitch))))

    uav = uav.Vehicle(x_init, u_init, wind_n, model, {}, state_space=state_space)

    Vr = uav.getAirspeed()
    AOA = uav.getAOA()
    SSA = uav.getSSA()
    R_sb = uav.getRotation_sb()
    s_omega = uav.getAngularVelocity(frame='s')
    quat = uav.getQuaternion_nb()
    WindGust = np.zeros([nSample,3])

    from uav.controllers.attitude_nmpc import set_data
    importlib.reload(set_data)
    data = set_data.set_data(Vr, aoa, ssa, quat, s_omega, model.P, state_space=state_space)
    data['model'] = model
    from uav.controllers.attitude_nmpc import set_opt
    importlib.reload(set_opt)
    opt = set_opt.set_opt(state_space=state_space)
    from uav.controllers.attitude_nmpc import Controller
    importlib.reload(Controller)

    from uav.controllers.attitude_nmpc import DC_Controller
    importlib.reload(DC_Controller)
    opt_nlp_sol = dict()

    ipopt_opt = {'print_level':0,\
                 'linear_solver':'ma97',\
                  'sb':'yes',\
                  'warm_start_init_point':'yes'}

    blocksqp_opt = {'warmstart':True,\
                    'jit':False,\
                    'print_iteration':False,\
                    'regularity_check':True,\
                    'warn_initial_bounds':False}#,\
                    # 'hessian_approximation':'exact'}

    # nlp_sol = dict()
    # nl_sol['solver'] = 'blocksqp'
    # opt_nlp_sol = {'jit'              : True,\
                    # 'bound_consistency' : True,\
                    # 'error_on_fail'     : True,\
                    # 'eval_errors_fatal' : True,\
                    # 'ipopt'             : ipopt_opt,\
                    # 'common_options'    : com_opt}

    if "jit" not in opt_nlp_sol:
        opt_nlp_sol["jit"] = True
    if "print_time" not in opt_nlp_sol:
        opt_nlp_sol["print_time"] = False
    # if "jit_options" not in opt_nlp_sol:
    #     opt_nlp_sol["jit_options"] = {"flags": "-O2"}

    for key, val in opt['OCP'].items(): data[key] = val

    # data['solver'] = 'sqpmethod'
    # data['solver'] = 'osqp'
    data['solver'] = 'blocksqp'
    data['solver_opt'] = blocksqp_opt
    data['nlpsol_opt'] = {'jit'         : True,\
                    'bound_consistency' : True,\
                    'error_on_fail'     : False,\
                    'eval_errors_fatal' : False,\
                    # 'ipopt'             : ipopt_opt,\
                    'blocksqp': blocksqp_opt,\
                    'common_options'    : {}}

    # controller = Controller.Controller(data, opt, opt_nlp_sol, state_space=state_space)
    data['d'] = 3
    controller = DC_Controller.Controller(data)

    V_ = np.zeros(nSample)
    E_V = np.zeros(nSample)

    K = range(nSample)
    K_mpc = K[0::round(fsSim/opt['fs'])]

    # Implement actuator contraint
    from uav.actuators import actu as act


    for k in K:
        print(k)
        if k in K_mpc:
            tSimStart = time.time()
            controller.update(uav, ref[:,k:])
            print("Elapsed time is {0:f}.".format(time.time() - tSimStart))
            if state_space == 'full':
                delta_a_ref = controller.u[0]
                delta_e_ref = controller.u[1]
                delta_t_ref = controller.u[2]
            elif state_space == 'longitudinal':
                delta_a_ref = 0
                delta_e_ref = controller.u[0]
                delta_t_ref = controller.u[1]
        if actuator_dynamic:
            u_ref = np.array([delta_a_ref, delta_e_ref, delta_t_ref])
            delta_a_ref, delta_e_ref, delta_t_ref = act.actuator(u_ref, uav.getControl().T, k, fsSim, T)

        uav.setAileronCmd(delta_a_ref)
        uav.setElevatorCmd(delta_e_ref)
        uav.setThrottleCmd(delta_t_ref)

        # Adding turbulence komponent
        Gust = np.zeros(3)
        Gust[0] = windgust[k,0] + wind_n[0]
        Gust[1] = windgust[k,1] + wind_n[1]
        Gust[2] = windgust[k,2] + wind_n[2]
        uav.setWind(Gust)

        X[:, k] = uav.getState().T
        U[:, k] = uav.getControl().T
        F[:, k] = uav.getForce().T
        M[:, k] = uav.getMoment().T
        ssa[k]     = uav.getSSA()
        aoa[k]     = uav.getAOA()
        v_r[k]     = uav.getAirspeed()
        WindGust[k,:] = Gust


        uav.update(t[k])
    # Variable processing___________________________________________________________________________________________________
    # Converting to euler angles:

    q = X[3:7][:]
    roll = np.zeros(len(t))
    pitch = np.zeros(len(t))
    yaw = np.zeros(len(t))

    ref_roll, ref_pitch, ref_yaw = ref_roll[0:len(t)], ref_pitch[0:len(t)], ref_yaw[0:len(t)]

    for i in range(len(t)):
        roll[i], pitch[i], yaw[i] = ng.rpy_quaternion(q[:,i])

    # Saving results
    results = dict()
    results['t'] = t
    results['X'] = X
    results['U'] = U
    results['F'] = F
    results['M'] = M
    results['ssa'] = ssa
    results['aoa'] = aoa
    results['v_r'] = v_r
    results['roll'] = roll
    results['pitch'] = pitch
    results['yaw'] = yaw
    results['ref_roll'] = ref_roll
    results['ref_pitch'] = ref_pitch
    results['ref_yaw'] = ref_yaw
    results['v_ref'] = ref_V
    results['WindGust'] = WindGust

    inte = str(zz)
    total['sim'+ inte] = results





# PLOTTING
#import PLOT
#PLOT(results, res2)












#fig, ax = plt.subplots()
#ax.plot(t, F.T)
#plt.show()
#fig, ax = plt.subplots()
#ax.plot(t, E_V)
#plt.show()

#inputFig, inputAxes = evaluation.plot.plotInput(t, U)
#stateFig, stateAxes = evaluation.plot.plotState(t, X)
# forceFig, forceAxes = evaluation.plot.plotForce(t, F, M)
# relFig, relAxes = evaluation.plot.plotRelativeVelocity(t, airspeed, aoa, ssa)

# stateAxes[0,1].plot(t, ref_roll*np.ones(nSample)*180/np.pi, linestyle='--')
#stateAxes[0,1].plot(t, ref_pitch[0:nSample]*180/np.pi, linestyle='--')
# stateAxes[0,1].plot(t, ref_yaw*np.ones(nSample)*180/np.pi, linestyle='--')

# fig, ax = plt.subplots()
# ax.plot(t, ref_V)
# ax.plot(t, V_*np.ones(nSample))
# plt.show(block=False)
