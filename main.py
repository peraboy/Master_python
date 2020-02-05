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

from lib.geometry import numpy_geometry as ng
from lib.trim import trim
import importlib
importlib.reload(uav)
#Simulation setup
tInit = 0.0
tFinal   = 50.0
dt = 0.01
fsSim = int(1/dt)
t = np.arange(tInit, tFinal, dt)
nSample = t.size
wind_n = np.array([-5.0, 0.0, 0.0]) 
model.Param = model.P #parameters.loadParameters('uav/models/gryteX8_param.mat')
model.Param['rudder_min'] = model.Param['rudder_max'] = 0

# Trim
Va = 18
R = np.inf
gamma = 0
res = trim.trim(Va, R, gamma, model)

x_trim = res[0]
u_trim = res[1]
x_init = x_trim
u_init = u_trim

# Reference
ref_roll = np.zeros(nSample)
ref_pitch = np.zeros(nSample)
ref_pitch[round(nSample/2):] = (20*np.pi/180)*np.ones(round(nSample/2))
ref_yaw = np.zeros(nSample)
ref_V = Va*np.ones(nSample)

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

uav = uav.Vehicle(x_init, u_init, wind_n, model, {}, state_space='longitudinal')

V_ = np.zeros(nSample)
E_V = np.zeros(nSample)
tSimStart = time.time()

for k in range(0, nSample):
    roll  = uav.getRollAngle()
    pitch = uav.getPitchAngle()
    yaw   = uav.getYawAngle()
    V     = uav.getGroundspeed()

    yawError = np.mod(ref_yaw[k] - uav.getYawAngle() + np.pi, 2*np.pi) - np.pi

    E_V[k] = ref_V[k] - V

    ref_roll[k]  = pidYaw.update(t[k], yawError)
    delta_a   = pidRoll.update(t[k], ref_roll[k] - roll)
    delta_e   = pidPitch.update(t[k], ref_pitch[k] - pitch)
    delta_t   = pidV.update(t[k], ref_V[k] - V)

    uav.setAileronCmd(delta_a)
    uav.setElevatorCmd(delta_e)
    uav.setThrottleCmd(delta_t)
    uav.setWind(wind_n)

    X[:, k] = uav.getState().T    
    U[:, k] = uav.getControl().T
    F[:, k] = uav.getForce().T
    M[:, k] = uav.getMoment().T

    airspeed[k] = uav.getAirspeed()
    aoa[k]      = uav.getAOA()
    ssa[k]      = uav.getSSA()
    V_[k] = V

    # acc[:,k] = uav.getSensorValue('acc').T
    # acc2[:,k] = uav.getSensorValue('acc2').T
    uav.update(t[k])
    
print("Elapsed time is {0:f}.".format(time.time() - tSimStart))

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

stateAxes[0,1].plot(t, ref_roll*np.ones(nSample)*180/np.pi, linestyle='--')
stateAxes[0,1].plot(t, ref_pitch*np.ones(nSample)*180/np.pi, linestyle='--')
stateAxes[0,1].plot(t, ref_yaw*np.ones(nSample)*180/np.pi, linestyle='--')

fig, ax = plt.subplots()
ax.plot(t, ref_V)
ax.plot(t, V_*np.ones(nSample))
plt.show(block=False)
