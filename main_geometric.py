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
import evaluation.plot

from lib.util import file_operations
from lib.geometry import casadi_geometry as cg
from lib.geometry import numpy_geometry as ng

from uav.controllers.geometric_controller.geometric_controller_functions import *
from uav.controllers.geometric_controller import controller as Geometric_controller

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
wind_n  = np.zeros(3)

model.Param = model.P

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

def is_valid_delta(delta, a, b):
    return delta < min([2-a-b, a-1, a+2*b-1])


model.P = model.Param
res = trim.trim(35, np.inf, 0.0, model)
x_trim = res[0]
Euler_trim = ng.rpy_quaternion(x_trim[3:7])

for tracking in [False]:
    f = 0.2
    if tracking:
        ref = design_reference(0, np.pi/12, 0.1, 0, np.pi/12, 0.1, t, run_test=False)
        Gamma_d     = ref[0]
        omega_d     = ref[1]
        omega_d_dot = ref[2]
        rollInit = 135.0*np.pi/180.0
        pitchInit = 0
        scenario = 'tracking'
    else:
        scenario = 'regulation'
        pitch_ref = Euler_trim[1]
        Gamma_d = Geometric_controller.compute_Gamma_rp(0, pitch_ref)
        Gamma_d = np.repeat(Gamma_d.reshape(3,1), nSample, axis=1)
        omega_d = np.zeros((3, nSample))
        omega_d_dot = np.zeros((3, nSample))

        rotation_axis = np.array((1,0,0))
        rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
        Gamma_init = -ng.rotation_axis_angle(rotation_axis, -1*np.pi/180)@Gamma_d[:,0]
        pitchInit = -np.arcsin(Gamma_init[0])#Euler_trim[1]
        rollInit = np.arctan2(Gamma_init[1],Gamma_init[2])

        

    file_appendix = '_' + scenario #'static_start'

    pitch_ref = -np.arcsin(Gamma_d[0,:])#Euler_trim[1]
    roll_ref = np.arctan2(Gamma_d[1,:],Gamma_d[2,:])
    pInit = 0.0

    qInit = 0.0

    yawInit = 0.0
    rInit = 0.0

    quatInit = ng.quaternion_rpy(rollInit, pitchInit, yawInit).reshape(4,1)
    linVelInit = np.vstack([35.0, 0.0, 0.0])
    angVelInit = np.vstack([pInit, qInit, rInit])

    xInit = np.vstack([posInit, quatInit, linVelInit, angVelInit])
    uInit = np.vstack([0.        , 0.0371, 0.        , 0.12193719])

    xInit = xInit.reshape(len(xInit))
    uInit = uInit.reshape(len(uInit))

    Gamma = Geometric_controller.compute_Gamma_rp(rollInit, pitchInit)
    pot = Geometric_controller.compute_nominal_potential(Gamma, Gamma_d[:,0])

    for option in range(0,1):
        tSimStart = time.time()
        uav = UAV.Vehicle(xInit, uInit, wind_n, model, sensors)

        U            = np.zeros([4, nSample])
        X            = np.zeros([13, nSample])
        XDOT         = np.zeros(X.shape)
        F            = np.zeros([3,nSample])
        M            = np.zeros([3,nSample])
        Omega        = np.zeros((nSample, 3))
        Euler        = np.zeros((nSample, 3))

        e_0        = np.zeros((3, nSample))
        e_1        = np.zeros((3 ,nSample))
        e_smoothed = np.zeros((3, nSample))
        e_Gamma = np.zeros((3, nSample))


        RollAngleRef = np.zeros(nSample)
        RollAngle = np.zeros(nSample)
        PitchAngle = np.zeros(nSample)
        airspeed     = np.zeros(nSample)
        aoa          = np.zeros(nSample)
        ssa          = np.zeros(nSample)
        Mode         = np.zeros(nSample)
        PSI_N        = np.zeros(nSample)
        PSI_E        = np.zeros(nSample)
        PSI_M        = np.zeros(nSample)
        Theta        = np.zeros(nSample)
        E_omega_norm = np.zeros(nSample)
        Delta_t = np.zeros(nSample)
        E_H = np.zeros(nSample)
        dynamic_mode = np.zeros(nSample)
        synergy_gap_smoothed = np.zeros(nSample)

        Omega_dot_old = np.zeros((3, nSample))
        Omega_dot_new = np.zeros((3, nSample))


        ## Controller Design
        Gamma = uav.getRotation_nb().T[:,2]

        a      = 1.25
        b      = 0.6
        delta  = 0.1
        kp     = 9.5
        kd     = 8.0
        Kd     = kd*np.eye(3)
        B      = 0
        Lambda = 0.09      #See Mayhew2013 6.5
        Phi    = 3/Lambda  #See Mayhew2013 6.5

        # Check conditions for exponential converence
        B_omega_d = 0
        eig_Kd = np.linalg.eig(Kd)
        min_eig_Kd = min(eig_Kd[0])
        max_eig_Kd = max(eig_Kd[0])

        epsilon_1 = np.sqrt(kp/max(1, b))
        epsilon_3 = 4*min_eig_Kd/(4*kp*max(1, abs(b)) + (max_eig_Kd + B_omega_d)**2)
        epsilon = min([epsilon_1, epsilon_3])
        B_e_omega = kp*delta/(epsilon*(1+abs(b)))

        ControlParam = {'a':a,\
                        'b':b,\
                        'kp':kp,\
                        'kd':kd,\
                        'Kd':Kd,\
                        'delta':delta,\
                        'P':P,\
                        'Q':np.diag((10,1,1)),\
                        'B_e_omega':B_e_omega,\
                        'c':epsilon,\
                        'Lambda':Lambda,\
                        'Phi':Phi,\
                        'J':P['I_cg'],\
                        'fs':fsSim,\
                        'option':option,\
                        'hybrid_control':option!=0,\
                        'hybrid_smoothed':False}


        print(is_valid_delta(delta, a, b))
        controller = Geometric_controller.Controller(uav, Gamma_d[:, 0], ControlParam)

        for k in range(0, nSample):

            uav.update(t[k])

            omega_dot_uav = uav.getDynamics()[10:].T

            controller.update(uav, Gamma_d[:, k], omega_d[:, k], omega_d_dot[:, k])
            control = controller.control

            delta_t = pidV.update(t[k], VRef[k] - uav.getGroundspeed())
            Delta_t[k] = delta_t

            uav.setAileronCmd(controller.control[0])
            uav.setElevatorCmd(controller.control[1])
            uav.setRudderCmd(controller.control[2])
            uav.setThrottleCmd(delta_t)
            uav.setWind(wind_n)

            Omega[k, :] = np.squeeze(uav.getAngularVelocity())
            XDOT[:,k]   = uav.getDynamics().T

            X[:, k]     = uav.getState().T
            U[:, k]     = uav.getControl().T
            F[:, k]     = uav.getForce().T
            airspeed[k] = uav.getAirspeed()
            aoa[k]      = uav.getAOA()
            ssa[k]      = uav.getSSA()

            # TODO: Write Logging option for controller
            Mode[k]  = controller.mode

            if ControlParam['hybrid_smoothed']:
                PSI_N[k] = controller.W[0]
                PSI_E[k] = controller.W[1]
                PSI_M[k] = controller.W[controller.mode]
            else:
                PSI_N[k] = controller.U[0]
                PSI_E[k] = controller.U[1]
                PSI_M[k] = controller.U[controller.mode]

            Theta[k] = controller.Theta
            E_omega_norm[k] = np.sqrt(np.dot(controller.eOmega, controller.eOmega))


            dynamic_mode[k] = controller.p
            synergy_gap_smoothed[k] = controller.synergy_gap_smoothed

            RollAngle[k] = uav.getRollAngle()
            PitchAngle[k] = uav.getPitchAngle()

            e_0[:, k]        = controller.e[0]
            e_1[:, k]        = controller.e[1]
            e_smoothed[:, k] = controller.e_smoothed
            e_Gamma[:, k] = controller.e_Gamma

        print("Elapsed time is {0:f}.".format(time.time() - tSimStart))


        Data = dict()
        Data['t'] = t
        Data['X'] = X
        Data['U'] = U
        Data['F'] = F
        Data['M'] = M
        Data['Mode'] = Mode
        Data['PSI_N'] = PSI_N
        Data['PSI_E'] = PSI_E
        Data['PSI_M'] = PSI_M
        Data['Theta'] = Theta
        Data['airspeed'] = airspeed
        Data['aoa'] = aoa
        Data['ssa'] = ssa
        Data['pitch_ref'] = pitch_ref
        Data['roll_ref'] = roll_ref
        Data['omega_d'] = omega_d
        Data['P'] = P

        # filename = 'results_option' + str(ControlParam['option']) + file_appendix
        # if option > 0 and ControlParam['hybrid_smoothed']:
        #     filename += '_smoothed'

        # file_operations.save_obj(Data, filename) 

fig, ax = evaluation.plot.plotState(t, X)
plt.show()

if False:
    fig, ax = plt.subplots(4,1)
    ax[0].plot(t, e_0.T)
    ax[1].plot(t, e_1.T)
    ax[2].plot(t, e_smoothed.T)
    ax[2].plot(t, e_Gamma.T)
    plt.show()


    fig, ax = plt.subplots()
    ax.plot(t, dynamic_mode)

    if False:
        rollAngleRef = np.zeros(nSample)
        pitchAngleRef = np.zeros(nSample)
        for k in range(0, nSample):
            pitchAngleRef[k] = np.arcsin(-Gamma_d[0,k])
            rollAngleRef[k] = np.arctan(Gamma_d[1,k]/Gamma_d[2,k])


        d = abs(abs(Theta) - np.pi/2)                                                                                 
        idx = np.where(d == np.min(d))

        fig, ax = plt.subplots(3,1)
        ax[0].plot(t, RollAngle)
        ax[0].plot(t, rollAngleRef)
        ax[0].plot(t[idx], RollAngle[idx], '*')
        ax[1].plot(t, PitchAngle)
        ax[1].plot(t, pitchAngleRef)
        ax[1].plot(t[idx], PitchAngle[idx], '*')
        ax[2].plot(t, d)

        ax[2].plot(t, abs(abs(Theta) - np.pi/2)) 

        for i in range(3):
            ax[i].grid(which='both', alpha = 0.5)

        fig, ax = plt.subplots(3,1)
        for i in range(3):
            ax[i].plot(t, U[i, :].T)

        fig, ax = plt.subplots()
        ax.plot(t, Omega_dot_old.T)
        ax.plot(t, Omega_dot_new.T)
        ax.set_xlabel('t')
        ax.set_ylabel('omega_dot')
        plt.show()

    # Export controller parameters to latex
    def write_latex_macro(file, name, value):
        value_string = "{0:.2f}".format(value)
        file.write("\\newcommand{\\" + name + "Sim}{" + value_string + "}\n")

    paper_dir = '../paper_hybrid/ifacconf_latex/'
    filepath = paper_dir + 'simulator_macros.sty'
    file = open(filepath, 'w')
    write_latex_macro(file, 'kp', ControlParam['kp'])
    write_latex_macro(file, 'kd', ControlParam['kd'])
    write_latex_macro(file, 'a', ControlParam['a'])
    write_latex_macro(file, 'b', ControlParam['b'])
    file.close()

# os.system("python plot_sim.py")
