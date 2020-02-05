import numpy as np
from lib.trim import trim as trim
from lib.geometry import numpy_geometry as ng

import uav.models.aerosonde as model
from uav import uav as UAV
from evaluation import plot

import matplotlib.pyplot as plt
import importlib
importlib.reload(trim)
importlib.reload(UAV)

# TODO: Implement a range of random tests
# TODO: Test with X8 model and Zagi

def test_trim(*args, **kwargs):
    tInit   = 0.0
    tFinal  = 5.0
    dt      = 0.01
    t       = np.arange(tInit, tFinal, dt)
    fsSim   = int(1/dt)
    N = t.size
    wind_n  = np.zeros(3)


    n_test = 10
    Va    = np.random.uniform(20            , 30           , n_test)
    # R     = np.inf(n_test) #
    R = np.random.uniform(100           , 200          , n_test)
    gamma = np.random.uniform(0*np.pi/180 , 20*np.pi/180 , n_test)
    # gamma = np.zeros(n_test) #np.random.uniform(-20*np.pi/180 , 20*np.pi/180 , n_test)

    pos = np.zeros(3)
    for i in range(n_test):
        model.Param = model.P
        res = trim.trim(Va[i], R[i], gamma[i], model, print_level=5)
        # res = trim.trim(35, np.inf, 0.0, model, print_level=5)
        # res = trim.trim(Va[i], np.inf, gamma[i], model, print_level=5)
        # res = trim.trim(Va[i], np.inf, 0.0, model, print_level=5)

        #Simulation setup
        # Euler = res[0][6:9]
        # vel   = res[0][3:6]
        # omega = res[0][9:]
        # u = res[1]
        # quat = ng.quaternion_rpy(Euler[0], Euler[1], Euler[2])
        # x = np.concatenate([pos, quat, vel, omega])
        x = res[0]
        u = res[1]

        uav = UAV.Vehicle(x, u, wind_n, model, {})

        X = np.zeros((13, N))
        for k in range(N):
            uav.update(t[k])
            X[:,k] = uav.getState()
            
        # print('Va: ' + str(Va[i]))
        # print('R: ' + str(R[i]))
        # print('gamma: ' + str(gamma[i]))

        print('Va - Va_d: ' + str(uav.getAirspeed() - Va[i]))
        print('gamma - gamma_d: ' + str(uav.getFlightPathAngle() - gamma[i]))
        plot.plotState(t, X)
        plt.show()
