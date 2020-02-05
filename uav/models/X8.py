import numpy as np 
from lib.geometry import numpy_geometry as ng


def compute_sigma(aoa, P):
    M = P['M']
    aoa_0 = P['alpha_0']
    sigma = (1+np.exp(-M*(aoa-aoa_0)) + np.exp(M*(aoa + aoa_0)))/\
            ((1+np.exp(-M*(aoa-aoa_0)))*(1+np.exp(M*(aoa + aoa_0))))
    return sigma

def compute_C_L_alpha(aoa, P):
    sigma = compute_sigma(aoa, P)
    C_L = (1 - sigma)*(P['C_L_0'] + P['C_L_alpha']*aoa) + sigma*2*np.sign(aoa)*np.sin(aoa)**2*np.cos(aoa)
    return C_L

def compute_C_D_alpha(aoa, P):
    C_D = P['C_D_p'] + (P['C_L_0'] + P['C_L_alpha']*aoa)**2/(np.pi*P['e']*P['AR'])
    return C_D

def dragForce(Vr, alpha, beta, q, delta_e, P, stall=False): #{{{2
    """ Returns the magnitude of the drag force in body-fixed frame."""    
    if stall:
        C_D_alpha = compute_C_D_alpha(alpha, P)
    else:
        C_D_alpha = P['C_D_0'] + P['C_D_alpha1']*alpha# + P['C_D_alpha2']*alpha**2

    D =  0.25 * P['S_wing'] * P['rho'] * Vr *\
            (P['C_D_q']*P['c']*q 
                + 2*Vr*(C_D_alpha + P['C_D_beta1']*beta + P['C_D_delta_e']*delta_e**2))
    return D

def liftForce(Vr, alpha, q, delta_e, P, stall=False):#{{{2
    """ Returns the magnitude of the lift force in body-fixed frame."""    
    # if stall:

    if stall:
        C_L_alpha = compute_C_L_alpha(alpha, P)
    else:
        C_L_alpha = P['C_L_0'] + P['C_L_alpha']*alpha

    L =  0.25 * P['S_wing'] * P['rho'] * Vr * (P['C_L_q']*P['c']*q \
            # + 2*Vr*(P['C_L_0'] + P['C_L_alpha']*alpha \
            + 2*Vr*(C_L_alpha + P['C_L_delta_e']*delta_e))
    return L

def sideForce(Vr, beta, p, r, delta_a, delta_r, P):#{{{2
    """ Returns the magnitude of the side force in body-fixed frame."""    
    Y =  0.25 * P['S_wing'] * P['rho'] * Vr *\
            (P['C_Y_p']*P['b']*p \
            + P['C_Y_r']*P['b']*r \
            + 2*Vr*(P['C_Y_0'] \
            + P['C_Y_beta']*beta \
            + P['C_Y_delta_a']*delta_a \
            + P['C_Y_delta_r']*delta_r))
    return Y

def rollMoment(Vr, beta, p, r, delta_a, delta_r, P):#{{{2
    """ Returns the magnitude of the roll moment in body-fixed frame."""    
    l =  0.25 * P['S_wing'] * P['rho'] * Vr * P['b'] * \
            (P['C_l_p']*P['b']*p \
            + P['C_l_r']*P['b']*r \
            + 2*Vr*(P['C_l_0'] \
                + P['C_l_beta']*beta \
                + P['C_l_delta_a']*delta_a \
                + P['C_l_delta_r']*delta_r))
    return l

def pitchMoment(Vr, alpha, q, delta_e, P):#{{{2
    """ Returns the magnitude of the pitch-moment in body-fixed frame."""    
    m =  0.25 * P['S_wing'] * P['rho'] * Vr * P['c'] * \
            (P['C_m_q']*P['c']*q \
            + 2*Vr*(P['C_m_0'] \
                + P['C_m_alpha']*alpha \
                + P['C_m_delta_e']*delta_e))
    return m

def yawMoment(Vr, beta, p, r, delta_a, delta_r, P):#{{{2
    """ Returns the magnitude of the yaw moment in body-fixed frame."""    
    n =  0.25 * P['S_wing'] * P['rho'] * Vr * P['b'] * \
            (P['C_n_p']*P['b']*p \
            + P['C_n_r']*P['b']*r \
            + 2*Vr*(P['C_n_0'] \
                + P['C_n_beta']*beta \
                + P['C_n_delta_a']*delta_a \
                + P['C_n_delta_r']*delta_r))
    return n

def thrustForce(Vr, delta_t, P):
    """ Returns the magnitude of the thrust-force in body-axes."""
        
    # Discharge velocity.
    Vd = Vr + delta_t*(P['k_motor']-Vr)
        
    # Beard2012/ Calculation of thrust in a ducted fan assembly for hovercraft
    T = 0.5*P['rho']*P['S_prop']*P['C_prop']*Vd*(Vd-Vr)
    return T

P = {'mass'         : 3.3640,\
     'Jxx'          : 1.2290,\
     'Jyy'          : 0.1702,\
     'Jzz'          : 0.8808,\
     'Jxz'          : 0.9343,\
     'S_wing'       : 0.7500,\
     'b'            : 2.1000,\
     'c'            : 0.3571,\
     'S_prop'       : 0.1018,\
     'k_motor'      : 40,\
     'k_T_p'        : 0,\
     'k_Omega'      : 0,\
     'C_prop'       : 1,\
     'C_L_alpha'    : 4.0203,\
     'C_L_0'        : 0.0867,\
     'C_L_q'        : 3.8700,\
     'C_L_delta_e'  : 0.2781,\
     'C_D_delta_e'  : 0.0633,\
     'C_D_alpha2'   : 1.0555,\
     'C_D_alpha1'   : 0.0791,\
     'C_D_0'        : 0.0197,\
     'C_D_beta2'    : 0.1478,\
     'C_D_beta1'    : -0.0058,\
     'C_D_q'        : 0,\
     'C_m_alpha'    : -0.4629,\
     'C_m_0'        : 0.0227,\
     'C_m_q'        : -1.3012,\
     'C_m_delta_e'  : -0.2292,\
     'C_Y_beta'     : -0.2239,\
     'C_Y_0'        : 0,\
     'C_Y_p'        : -0.1374,\
     'C_Y_r'        : 0.0839,\
     'C_Y_delta_a'  : 0.0433,\
     'C_Y_delta_r'  : 0,\
     'C_l_beta'     : -0.0849,\
     'C_l_0'        : 0,\
     'C_l_p'        : -0.4042,\
     'C_l_r'        : 0.0555,\
     'C_l_delta_a'  : 0.1202,\
     'C_l_delta_r'  : 0,\
     'C_n_beta'     : 0.0283,\
     'C_n_0'        : 0,\
     'C_n_p'        : 0.0044,\
     'C_n_r'        : -0.0720,\
     'C_n_delta_a'  : -0.0034,\
     'C_n_delta_r'  : 0,\
     'rho'          : 1.2250,\
     'gravity'      : 9.8100,\
     'aileron_min'  : -0.6109,\
     'aileron_max'  : 0.6109,\
     'elevator_min' : -0.6109,\
     'elevator_max' : 0.6109,\
     'rudder_min'   : 0,\
     'rudder_max'   : 0,\
     'throttle_min' : 0,\
     'throttle_max' : 1}


P['r_cg'] = np.zeros(3)
P['I_cg'] = np.array([[P['Jxx'],0,-P['Jxz']], [0, P['Jyy'], 0], [-P['Jxz'], 0, P['Jzz']] ])
P['M_rb'] = np.vstack([np.hstack([np.eye(3) * P['mass'], -P['mass'] * ng.Smtrx(P['r_cg'])]),
                       np.hstack([P['mass'] * ng.Smtrx(P['r_cg']), P['I_cg']])])
P['M_rb_inv'] = np.linalg.inv(P['M_rb'])
P['J'] = P['I_cg']
P['AR'] = P['b']**2/P['S_wing']
P['e'] = 0.9
P['M']           = 50
P['alpha_0']     = 0.4712
P['epsilon']     = 0.1592









