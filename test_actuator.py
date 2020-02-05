import numpy as np
import importlib
from uav.actuators import actuator #import Actuator as Actuator
importlib.reload(actuator)
# import uav.actuators.Actuator as Actuator
import matplotlib.pyplot as plt

fs = 100
dt = 1/fs
t_end = 3
t_0 = 0
t = np.arange(t_0, t_end, dt)
N = t.shape[0]
Ref = np.zeros(N)
Val = np.zeros(N)
Ref[int(N/2):] = np.ones(Ref[int(N/2):].shape)

val = 0
T = 0.1
lb = -10
ub =  10

val = Val[1]
ref = Ref[1]
actuator = actuator.Actuator(val, ref, T, fs, lb, ub)
for i in range(1,N):
    actuator.set_reference(Ref[i])
    actuator.update()
    Val[i] = actuator.get_state()
    
fig, ax = plt.subplots()
ax.plot(t, Ref)
ax.plot(t, Val)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Actuator State [-]')
plt.show()
# xlabel('Time [s]')
# ylabel('output [-]')
