import numpy as np
from lib.geometry import numpy_geometry as ng

def set_data(Vr, alpha, beta, quat, s_omega, P, state_space='full'):
    """TODO: Docstring for set_data.

    :Vr: TODO
    :alpha: TODO
    :beta: TODO
    :quat_init: TODO
    :s_omega: TODO
    :returns: TODO

    """
    data = dict()
    if state_space == 'full':
        data['u_init'] = [0.0, 0.0, 0.5]
        data['u_min']  = [P['aileron_min'], P['elevator_min'] , P['throttle_min']]
        data['u_max']  = [P['aileron_max'], P['elevator_max'] , P['throttle_max']]
        # data['x_init'] = [Vr,  beta, alpha,] + quat.tolist() + s_omega.tolist()
        data['x_init'] = [18,  0, 0, 1, 0, 0, 0, 0, 0, 0] # + quat.tolist() + s_omega.tolist()
        # data['x_min']  = [1.0 , -np.pi , -np.pi/2 , -1 , -1 , -1 , -1 , -np.inf, -np.inf , -np.inf]
        # data['x_max']  = [30  , +np.pi , +np.pi/2 , +1 , +1 , +1 , +1 , +np.inf, +np.inf , +np.inf]
        data['x_min']  = [1.0 , -np.pi , -np.pi/2 , -1 , -1 , -1 , -1 , -np.pi, -np.pi , -np.pi]
        data['x_max']  = [30  , +np.pi , +np.pi/2 , +1 , +1 , +1 , +1 , +np.pi, +np.pi , +np.pi]
    elif state_space == 'longitudinal':
        pitch = ng.rpy_quaternion(quat)[1]
        q = s_omega[1]
        data['u_init'] = [0.0, 0.5]
        data['u_min']  = [P['elevator_min'] , P['throttle_min']]
        data['u_max']  = [P['elevator_max'] , P['throttle_max']]
        data['x_init'] = [Vr  , alpha , pitch, q]
        data['x_min']  = [1.0 , -np.pi/2 , -np.pi/2     , -np.pi]
        data['x_max']  = [30  , +np.pi/2 , +np.pi/2     , +np.pi]
    elif state_space == 'lateral':
        print('Not implemented state_space: ' + state_space)
    else:
        print('Unknown state_space: ' + state_space)
        
    data['Param'] = P

    return data

