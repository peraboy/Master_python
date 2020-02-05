import numpy as np

def thrustForceCoates(airspeed, angularVelocity):
    """TODO: Docstring for thrustForceCoates.

    :airspeed: TODO
    :angularVelocity: TODO
    :returns: TODO

    """
    # Advance ratio.
    J = 2.0*np.pi*airspeed/(angularVelocity*P['D'])

    # Thrust coefficient
    C_T = P['C_T_0'] + P['C_T_J']*J

    # Thrust force.
    T = (P['rho']*D**4 *C_T/(4*np.pi**2)) * angularVelocity**2 * angularVelocity**2

