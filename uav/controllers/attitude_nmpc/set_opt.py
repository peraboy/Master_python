import numpy as np
def set_opt(state_space='full'):
    """TODO: Docstring for set_opt.

    :*args: TODO
    :**kwargs: TODO
    :returns: TODO

    """
        
    opt = dict()
    opt['OCP'] = dict()

    fs = 20
    N = 40
    T = 10

    if state_space == 'full':
        # Weighting matrices
        # No weighting on the aoa???
        Q_Vr     = np.array(1)
        Q_qDot   = np.array(10*1e1)
        Q_qCross = np.sqrt(10)# * np.ones(3)
        Q_omegaS = 1# * np.ones(3)
        Q_phi    = np.array(1)
        Q_beta   = np.array(1)

        # Q = np.diag(np.concatenate((Q_Vr, Q_qDot, Q_qCross, Q_omegaS, Q_phi, Q_beta)))
        Q = np.diag(np.array((Q_Vr, Q_qDot, Q_qCross, Q_qCross, Q_qCross, Q_omegaS, Q_omegaS, Q_omegaS, Q_phi, Q_beta)))

        Q_N = Q
        Q_dU = 1e-2*np.eye(3)  * 1/(T/N)**2

        R_delta_a = 1e-3
        R_delta_e = 1e-3
        R_delta_t = 1e-3

        R = np.diag((R_delta_a, R_delta_e, R_delta_t))
    elif state_space == 'longitudinal':
            q_Vr = 1
            q_alpha  = 1
            q_theta = 1
            q_q = 1e-3
            Q = np.diag((q_Vr, q_alpha, q_theta, q_q))
            Q_N = Q

            r_delta_e = 1e-3
            r_delta_t = 1e-3
            R = np.diag((r_delta_e, r_delta_t))

            Q_dU = 1e-2*np.eye(2) * 1/(T/N)**2 

    elif state_space == 'lateral':
        print('Not implemented state_space: ' + state_space)
    else:
        print('Unknown state_space: ' + state_space)


    opt['fs'] = fs # Update rate
    opt['OCP']['N'] = N # Number of control intervals
    opt['OCP']['T'] = T # Prediction horizon

    # Weight State
    opt['OCP']['Q']      = Q
    # Weight Final state
    opt['OCP']['Q_N']    = Q_N
    # Weight Control variable difference
    opt['OCP']['Q_dU']   = Q_dU
    # Weight Control variable
    opt['OCP']['R']      = R
    # Weight Slack variable
    opt['OCP']['W']      = [1.0, 1.0, 1e4, 1e4]
    # Backoff parameter
    opt['OCP']['eps']    = 0.3

    return opt
