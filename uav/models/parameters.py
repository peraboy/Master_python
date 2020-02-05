import numpy as np
import scipy.io as sio
from lib.geometry import numpy_geometry as ng

def loadParameters(fileName):
    P = sio.loadmat(fileName)
    del P['__header__'], P['__version__'], P['__globals__']

    for key, value in P.items():
        if P[key].size == 1:
            P[key] = P[key][0][0]

    P['r_cg'] = np.zeros(3)

    
    P["rho"] = 1.225
    P['gravity'] = 9.81
    P['I_cg'] = np.array([[P['Jx'],0,-P['Jxz']], [0, P['Jy'], 0], [-P['Jxz'], 0, P['Jz']] ])
    P['M_rb'] = np.vstack([np.hstack([np.eye(3) * P['mass'], -P['mass'] * ng.Smtrx(P['r_cg'])]),
                           np.hstack([P['mass'] * ng.Smtrx(P['r_cg']), P['I_cg']])])

    P['aileron_min'] = -np.pi/3.0
    P['aileron_min']  = -np.pi/3.0
    P['aileron_max']  = +np.pi/3.0
    P['elevator_min'] = -np.pi/3.0
    P['elevator_max'] = +np.pi/3.0
    P['throttle_min'] = 0.0
    P['throttle_max'] = 1.0

    P['kp_h']     = -0.025
    P['ki_h']     =  0.0

    P['kp_theta'] = 0.1
    P['kd_theta'] = -0.01
    P['ki_theta'] = 0

    P['kp_V']     = -0.05
    P['ki_V']     = -0.01

    P['kp_phi']   = -0.5
    P['ki_phi']   = 0
    P['kd_phi']   = 0

    P['kp_chi'] = -0.05
    P['ki_chi'] = 0.0

    # Sensor parameters
    P['stdAcc'] = 1e-2
    P['stdGyr'] = 1e-2
    P['stdSP'] = 1e-2
    P['stdDP'] = 1e-2
    P['stdMag'] = 1e-2

    gam = P['Jx']*P['Jz'] - P['Jxz']*P['Jxz']
    P['Gamma'] = [gam, \
            P['Jxz']*(P['Jx']-P['Jy']+P['Jz'])/gam, \
            (P['Jz']*(P['Jz']-P['Jy'])+P['Jxz']*P['Jxz'])/gam,\
             P['Jz']/gam,\
             P['Jxz']/gam,\
             (P['Jz']-P['Jx'])/P['Jy'],\
             P['Jxz']/P['Jy'],\
             ((P['Jx']-P['Jy'])*P['Jx']+P['Jxz']*P['Jxz'])/gam,\
             P['Jx']/gam]
    return P
