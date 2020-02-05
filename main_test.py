import numpy as np
import scipy.io as sio
import time
import matplotlib.pyplot as plt
import importlib
import os as os

from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs

from uav.controllers import PID
from uav.models import parameters
import uav.models.aerosonde as model
import uav.sensors.accelerometer as accelerometer
from uav import uav as UAV
# import uav.models.gryteX8 as model

import evaluation.plot

from lib.util import file_operations
from lib.geometry import casadi_geometry as cg
from lib.geometry import numpy_geometry as ng
# from lib.geometric_controller.geometric_controller_functions import *
from uav.controllers.geometric_controller import controller as Geometric_controller

from test_reference_design import design_reference
from test_reference_design import design_reference_analytic
importlib.reload(Geometric_controller)
importlib.reload(UAV)

from lib.trim import trim

#Sensors
sensors = dict()
acc_bias = [1,2,3]
acc_std = [0,0,0]
sensors['acc'] = accelerometer.Accelerometer(bias=acc_bias,std=acc_std)

acc_bias2 = [0,0,0]
acc_std2 = [1,2,3]
sensors['acc2'] = accelerometer.Accelerometer(bias=acc_bias2,std=acc_std2)

#Simulation setup
tInit   = 0.0
tFinal  = 10.0
dt      = 0.01
t       = np.arange(tInit, tFinal, dt)
fsSim   = int(1/dt)
nSample = t.size

model.Param = model.getParameters()

# Set artificially high actuator bounds
model.Param['aileron_min']  = -np.pi
model.Param['aileron_max']  = +np.pi
model.Param['elevator_min'] = -np.pi
model.Param['elevator_max'] = +np.pi
model.Param['rudder_min']   = -np.pi
model.Param['rudder_max']   = +np.pi

# Reference
VRef          = 35 * np.ones((nSample, 1))
yawAngleRef   = 0  * (np.pi/180.0) * np.ones((nSample, 1))
pitchAngleRef = 20 * (np.pi/180.0) * np.ones((nSample, 1))
Ref = np.hstack([VRef, yawAngleRef.reshape(nSample, 1), pitchAngleRef])

# Initial state
posInit = np.vstack([0.0, 0.0, -200.0])

P = model.Param

pidRoll  = PID(0.78  , 0.01  , -0.11 , -0.09 , 0.09 , 0.1  , P['aileron_min']  , P['aileron_max'])
pidPitch = PID(-0.78 , -0.30 , -0.16 , -0.1  , 0.1  , 0.1  , P['elevator_min'] , P['elevator_max'])
pidYaw   = PID(1.08  , 0.036 , 0.0   , -0.09 , 0.09 , 0.1)
pidV     = PID(0.69  , 10.0  , 0.0   , -0.09 , 0.09 , 1.0  , P['throttle_min'] , P['throttle_max'])


model.P     = model.Param
P = model.P

J = P['I_cg']
Jinv = np.linalg.inv(J)

n_test = int(1e3)

E = {'tau':np.zeros((3, n_test)),\
        # 'omega_dot':np.zeros((3, n_test))}
     'drift_vector':np.zeros((3, n_test)),\
     'control_effectiveness_matrix':np.zeros((1, n_test)),\
     'damping_matrix':np.zeros((1, n_test)),\
     'coriolis_matrix':np.zeros((1, n_test))}
# E[0] ,STR[0]= np.zeros((3, n_test))
# e_tau = np.zeros((3, n_test))
# e_omega_dot = np.zeros((3, n_test))
for i in range(n_test):
    pos_init    = np.random.rand(3)
    quat_init   = np.random.rand(4)
    linvel_init = np.random.rand(3)
    angvel_init = np.random.rand(3)
    # angvel_init = np.zeros(3) # np.random.rand(3)
    wind_n      = np.random.rand(3)
    u_init      = np.random.rand(4)

    quat_init = quat_init/np.linalg.norm(quat_init)

    x_init = np.concatenate([pos_init, quat_init, linvel_init, angvel_init])


    uav = UAV.Vehicle(x_init, u_init, wind_n, model, sensors)
    # uav.update(0.0)

    quat = uav.getQuaternion_nb()
    Vr = uav.getAirspeed()
    aoa = uav.getAOA()
    ssa = uav.getSSA()
    omega = uav.getAngularVelocity()
    v_b = uav.getLinearVelocity()
    u = uav.getControl()
    tau = UAV.compute_force(t, quat, Vr, aoa, ssa, omega, u, model)[3:]


    f = UAV.compute_f( omega, UAV.driftVector(Vr, aoa, ssa, P), UAV.dampingMatrix(Vr, P), Jinv)
    G = UAV.compute_G( UAV.controlEffectivenesMatrix(Vr, P), Jinv)
    E['tau'][:,i] = f + G@u[:3] - tau

    # Do not work anymore, because moved function
    # E['omega_dot'][:,i] = Jinv@(np.cross(J@omega,omega) + f + G@u[0:3]) - uav.angVel_b_dot_old

    # E['drift_vector'][:,i] = driftVector(Vr, aoa, ssa, P) - uav.driftVector()
    # E['control_effectiveness_matrix'][:,i] = 3 - np.trace(np.linalg.inv(controlEffectivenesMatrix(Vr, P))@uav.controlEffectivenesMatrix())
    # E['damping_matrix'][:,i] = 3 - np.trace(np.linalg.inv(dampingMatrix(Vr, P))@uav.dampingMatrix())
    # E['coriolis_matrix'][:,i] = 3 - np.trace(np.linalg.inv(getCoriolisMatrix(omega, P))@uav.getCoriolisMatrix())
    # G = uav.getCoriolisMatrix()
    # E['coriolis_matrix'][:,i] = np.sum(getCoriolisMatrix(omega, P) - uav.getCoriolisMatrix()) 

e = {}
for key in E.keys():
    e[key] = 0
    for i in range(E[key].shape[0]):
        e[key] += E[key][:,i]@E[key][:,i].T

for key, val in e.items():
    print(key + ': ', str(e[key]))
# e = np.zeros(2)
# for i in range(3):
#     e[0] += e_tau[i,:]@e_tau[i,:].T
#     e[1] += e_omega_dot[i,:]@e_tau[i,:].T

# print('e_tau:' + str(e[0]))
# print('e_omega_dot:' + str(e[1]))




