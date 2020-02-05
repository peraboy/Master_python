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


importlib.reload(trim)
importlib.reload(uav)
#Simulation setup
tInit = 0.0
tFinal   = 10.0
dt = 0.01
fsSim = int(1/dt)
t = np.arange(tInit, tFinal, dt)
nSample = t.size
N = nSample
wind_n = np.array([0.0, 0.0, 0.0]) 
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
P = model.P
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

state_space = 'full'

# Reference
ref_roll = np.zeros(nSample)
ref_pitch = np.zeros(nSample)
# ref_pitch[round(nSample/2):] = (20*np.pi/180)*np.ones(round(nSample/2))
ref_yaw = np.zeros(nSample)
ref_V   = Va*np.ones(nSample)

# Add more to include prediction horizon
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
aoa = uav.getAOA()
ssa = uav.getSSA()
R_sb = uav.getRotation_sb()
s_omega = uav.getAngularVelocity(frame='s')
quat = uav.getQuaternion_nb()

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
              # 'linear_solver':'ma97',\
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

    uav.setAileronCmd(delta_a_ref)
    uav.setElevatorCmd(delta_e_ref)
    uav.setThrottleCmd(delta_t_ref)
    uav.setWind(wind_n)

    X[:, k] = uav.getState().T    
    U[:, k] = uav.getControl().T
    F[:, k] = uav.getForce().T
    M[:, k] = uav.getMoment().T

    uav.update(t[k])
    

fig, ax = plt.subplots()
ax.plot(t, F.T)
plt.show()
fig, ax = plt.subplots()
ax.plot(t, E_V)
plt.show()

inputFig, inputAxes = evaluation.plot.plotInput(t, U)
stateFig, stateAxes = evaluation.plot.plotState(t, X)
# forceFig, forceAxes = evaluation.plot.plotForce(t, F, M)
# relFig, relAxes = evaluation.plot.plotRelativeVelocity(t, airspeed, aoa, ssa)

# stateAxes[0,1].plot(t, ref_roll*np.ones(nSample)*180/np.pi, linestyle='--')
stateAxes[0,1].plot(t, ref_pitch[0:nSample]*180/np.pi, linestyle='--')
# stateAxes[0,1].plot(t, ref_yaw*np.ones(nSample)*180/np.pi, linestyle='--')

# fig, ax = plt.subplots()
# ax.plot(t, ref_V)
# ax.plot(t, V_*np.ones(nSample))
plt.show(block=False)
