import numpy as np
import uav.models.X8 as model
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
start = 0
step = 0.01
number = 35
alpha = np.zeros(number)
for k in range(number):
    alpha[k] = start
    start += step
#________________________________________

#________________________________________

Cl = np.zeros(len(alpha))
Cd = np.zeros(len(alpha))
Cl_red = np.zeros(len(alpha))
Cd_red = np.zeros(len(alpha))
P = model.P

for i in range(len(alpha)):
    Cl[i] = P['C_L_0'] + P['C_L_alpha'] * alpha[i]
    Cd[i] = P['C_D_0'] + P['C_D_alpha1'] * alpha[i]

    Cl_red[i] = (P['C_L_0'] + (-0.1)) + (P['C_L_alpha'] + (-0.2) ) * alpha[i]
    Cd_red[i] = (P['C_D_0'] + (0.05)) + (P['C_D_alpha1'] + (2*alpha[i]) ) * alpha[i]

alpha = alpha* 180/np.pi

fig, axs = plt.subplots(2,1)
axs[0].plot(alpha, Cl, alpha, Cl_red, '--')
axs[0].set_title('Aerodynamic Coefficients')
#axs[0].set_xlim()
axs[0].set_ylabel('Cl')
axs[0].grid(True)

axs[1].plot(alpha, Cd, alpha, Cd_red, '--')
#axs[1].set_xlim()
axs[1].set_ylabel('Cd')
axs[1].grid(True)

