import numpy as np
from lib.geometry import numpy_geometry as ng
from scipy.integrate import ode

def dampingMatrix(Vr, P):
    # Return damping matrix of rotational dynamics.
    D = np.zeros((3,3))
    D[0,0] = P['b']**2*P['C_l_p']
    D[0,2] = P['b']**2*P['C_l_r']
    D[1,1] = P['c']**2*P['C_m_q']
    D[2,0] = P['b']**2*P['C_n_p']
    D[2,2] = P['b']**2*P['C_n_r']
    D *= 0.25*P['rho']*P['S_wing']*Vr

    return D

def controlEffectivenesMatrix(Vr, P):
    # Return control effectivenes matrix of rotational dynamics.
    G = np.zeros((3,3))
    G[0,0] = P['b']*P['C_l_delta_a']
    G[0,2] = P['b']*P['C_l_delta_r']
    G[1,1] = P['c']*P['C_m_delta_e']
    G[2,0] = P['b']*P['C_n_delta_a']
    G[2,2] = P['b']*P['C_n_delta_r']
    G *= 0.5*P['rho']*P['S_wing']*Vr**2
    
    return G

def driftVector(Vr, aoa, ssa, P):
    # Return drift vector of rotational dynamics.
    f = np.zeros(3)
    f[0] = P['b']*(P['C_l_0'] + P['C_l_beta']*ssa)
    f[1] = P['c']*(P['C_m_0'] + P['C_m_alpha']*aoa)
    f[2] = P['b']*(P['C_n_0'] + P['C_n_beta']*ssa)
    f *= 0.5*P['rho']*P['S_wing']*Vr**2
    return f

def getCoriolisMatrix(angVel_b, P):
    """TODO: Docstring for getCoriolisMatrix.
    :returns: TODO

    """
    c_rb_11 =  P['mass']*ng.Smtrx(angVel_b)
    c_rb_12 = -P['mass']*ng.Smtrx(angVel_b)  @ ng.Smtrx(P['r_cg'])
    c_rb_21 =  P['mass']*ng.Smtrx(P['r_cg']) @ ng.Smtrx(angVel_b)
    c_rb_22 = -ng.Smtrx(P['I_cg']@angVel_b)

    C_rb = np.vstack([np.hstack([c_rb_11, c_rb_12]),\
                      np.hstack([c_rb_21, c_rb_22])])
    return C_rb

def compute_f( omega, drift_force, damping_matrix, Jinv):
    return drift_force + damping_matrix@omega

def compute_G(controlEffectivenesMatrix, Jinv):
    return controlEffectivenesMatrix

def compute_force(t, quat, Vr, alpha, beta, b_omega, u, model, state_space='full'):
    """ Returns a stacked vector [F_b, M_b] of total force & moment vector in R^3 in body-fixed axes. """

    p = b_omega[0]
    q = b_omega[1]
    r = b_omega[2]
    delta_a = u[0]
    delta_e = u[1]
    delta_r = u[2]
    delta_t = u[3]

    P = model.Param

    if state_space == 'full':
        D = model.dragForce(Vr, alpha, beta, q, delta_e, P)
        Y = model.sideForce(Vr, beta, p, r, delta_a, delta_r, P)
        L = model.liftForce(Vr, alpha, q, delta_e, P)
        T = model.thrustForce(Vr, delta_t, P)
        G = P['mass'] * P['gravity']
        l = model.rollMoment(Vr, beta, p, r, delta_a, delta_r, P)
        m = model.pitchMoment(Vr, alpha, q, delta_e, P)
        n = model.yawMoment(Vr, beta, p, r, delta_a, delta_r, P)
    elif state_space == 'longitudinal':
        D = model.dragForce(Vr, alpha, beta, q, delta_e, P)
        Y = 0.0
        L = model.liftForce(Vr, alpha, q, delta_e, P)
        T = model.thrustForce(Vr, delta_t, P)
        l = 0.0
        m = model.pitchMoment(Vr, alpha, q, delta_e, P)
        n = 0.0
    elif state_space == 'lateral':
        D = 0.0
        Y = model.sideForce(Vr, beta, p, r, delta_a, delta_r, P)
        L = 0.0
        T = 0.0
        G = 0.0
        l = model.rollMoment(Vr, beta, p, r, delta_a, delta_r, P)
        m = 0.0
        n = model.yawMoment(Vr, beta, p, r, delta_a, delta_r, P)
    else:
        #TODO: Make this a proper error message.
        print('Unknown state_space')

    G = P['mass'] * P['gravity']

    # Aerodynamic force vector in body-fixed axes components.
    R_wb = ng.rotation_wb(alpha, beta)
    Fa_b = R_wb.T @ np.array((-D, Y, -L))

    # Propulsion force vector in body-fixed axes components.
    Ft_b = np.array((T, 0, 0))

    # Gravity force vector in body-fixed axes components.
    R_nb = ng.rotation_quaternion(quat)
    Fg_b = R_nb.T @ np.array((0, 0 , G))

    # Total force vector in body-fixed axes c
    F  = Fa_b + Ft_b + Fg_b

    # Compute aerodynamic moment vector in body-fixed axes

    M = np.array((l, m, n))

    tau = np.concatenate((F, M))
    return tau

def dynamics(t, x, tau, P):
    """ Returns the state dynamics based on state x = [p_n, quat_nb, vel_b, angVel_b] and tau = [F_b, M_b]."""
    
    quat_nb  = x[3:7]
    vel_b    = x[7:10]
    angVel_b = x[10:]
    
    C_rb = getCoriolisMatrix(angVel_b, P)
    # Fossen eq. 3.42
    ny = np.concatenate((vel_b, angVel_b))
    ny_dot = P['M_rb_inv']@ (tau - C_rb@ny)
    
    pos_n_dot = ng.rotation_quaternion_2(quat_nb)@vel_b
    quat_nb_dot = ng.quaternion_dot(quat_nb, angVel_b)
    vel_b_dot = ny_dot[0:3]
    angVel_b_dot = ny_dot[3:]

    xdot = np.concatenate([pos_n_dot, quat_nb_dot, vel_b_dot, angVel_b_dot])
    return xdot

class Vehicle(object):

    """Docstring for Vehicle. """

    def __init__(self, state, control, wind_n, model, sensors, t=0, state_space='full'):
        """TODO: to be defined1.

        :position: TODO
        :quaternion_nb: TODO
        :linearVelocity_b: TODO
        :angularVelocity_b: TODO
        :aileron: TODO
        :elevator: TODO
        :rudder: TODO
        :throttle: TODO
        """
        self.state_space = state_space

        self.aileron = control[0]
        self.elevator = control[1]
        self.rudder = control[2]
        self.throttle = control[3]

        self.t = t
        self.state = state
        self.xdot = np.zeros(13)

        self.solver = ode(dynamics)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(self.state, self.t)
        
        self.model = model
        self.model.Param['M_rb_inv'] = np.linalg.inv(self.model.Param['M_rb'])

        self.aileronCmd = self.aileron
        self.elevatorCmd = self.elevator
        self.rudderCmd = self.rudder
        self.throttleCmd = self.throttle

        self.wind_n = wind_n
        self.sensors = sensors
        self.sensorValue = dict()
        
        self.updateForce()
        self.updateSensors()


    def update(self, t):
        """Updates the vehicle class, including control inputs, force vector, state vector and sensors .
        :returns: 

        """
        self.t = t
        self.updateAileron()
        self.updateElevator()
        self.updateRudder()
        self.updateThrottle()
        self.updateForce()
        self.updateState()
        self.updateSensors()
        return
    

    def getAirspeed(self):
        """ Computes airspeed as Euclidean norm of the relative velocity vector.
        :returns: [m/s] Airspeed 

        """
        return np.linalg.norm(self.getRelativeVelocity)

    def updateSensors(self):
        """Updates all sensors in the list.
        :returns: 

        """
        for s in self.sensors:
            self.sensorValue[s] = self.sensors[s].update(self)
        return
    
    def updateState(self):
        """Updates the state of the vehicle.
        :returns: 

        """

        self.solver.set_f_params(self.tau, self.model.Param)
        self.solver.integrate(self.t)
        self.state = self.solver.y
        self.state[3:7] = self.state[3:7]/np.linalg.norm(self.state[3:7])
        return


    def updateAileron(self):
        """Updates the aileron deflection.
        :returns:

        """
        self.aileron = np.clip(self.aileronCmd, self.model.Param['aileron_min'], self.model.Param['aileron_max']) 
        return

    def updateElevator(self):
        """Updates the elevator deflection.
        :returns:

        """
        self.elevator = np.clip(self.elevatorCmd, self.model.Param['elevator_min'], self.model.Param['elevator_max']) 
        return

    def updateRudder(self):
        """Updates the rudder deflection.
        :returns:

        """
        self.rudder = np.clip(self.rudderCmd, self.model.Param['rudder_min'], self.model.Param['rudder_max']) 
        return

    def updateThrottle(self):
        """Updates the throttle position.
        :returns:

        """
        self.throttle = np.clip(self.throttleCmd, self.model.Param['throttle_min'], self.model.Param['throttle_max']) 
        return
    
    def updateForce(self):
        """Updates the fore vector.
        :returns:

        """
        quat  = self.getQuaternion_nb()
        omega = self.getAngularVelocity()
        Vr    = self.getAirspeed()
        alpha = self.getAOA()
        beta  = self.getSSA()

        self.tau = compute_force(0.0, quat, Vr, alpha, beta, omega, self.getControl(), self.model, self.state_space)

        return


    def getDynamics(self):
        """Computes the time-derivative of the state vector.
        :returns: xdot = [p_dot, quat_dot, linvel_dot, angvel_dot]

        """
        return self.xdot

    def getPosition(self, frame='n'):
        """Returns the position with respect to {n}.
        :returns: pos_n

        """
        return self.state[0:3]
    
    def getQuaternion_nb(self):
        """Returns the attitude quat_nb that describes the orientation of {b} wrt. {n}.
        :returns: TODO

        """
        return self.state[3:7]

    def getRotation_nb(self):
        """Returns the attitude rotation matrix R_nb that describes the orientation of {b} wrt. {n}, i.e. the columns of R_nb are the axes of the vehicle in {n}.
        :returns: R_nb

        """
        return ng.rotation_quaternion(self.getQuaternion_nb())

    def getRotation_wb(self):
        """Returns the orientation of {b} wrt. {w}
        :returns: R_wb

        """
        return ng.rotation_wb(self.getAOA(), self.getSSA())

    def getRotation_sb(self):
        """Return the orientation of {b} wrt. {s}
        :returns: R_sb

        """
        return ng.rotation_sb(self.getAOA())

    def getRPY_nb(self):
        """Returns Euler angles roll, pitch and yaw [rad] in intrinsic z-y-x convention. The angles describe the orientation of {b} wrt. {n}.
        :returns: roll, pitch, yaw

        """
        return ng.rpy_quaternion(self.getQuaternion_nb())

    def getRollAngle(self):
        """Returns the roll angle in [rad].
        :returns: roll angle [rad]

        """
        return ng.rollAngle_quaternion(self.getQuaternion_nb())

    def getPitchAngle(self):
        """Returns the pitch angle in [rad].
        :returns: pitch angle [rad]

        """
        return ng.pitchAngle_quaternion(self.getQuaternion_nb())

    def getYawAngle(self):
        """Returns the yaw angle in [rad].
        :returns: yaw angle [rad]

        """
        return ng.yawAngle_quaternion(self.getQuaternion_nb())

    def getLinearVelocity(self, frame='b'):
        """Returns the linear velocity vector in either {b} or {n}. Default is {b}.

        :frame: Frame in which the linear velocity vector is requested.
        :returns: v_{frame}

        """
        if frame == 'b':
            return self.state[7:10]
        elif frame == 'n':
            return self.getRotation_nb()@self.state[7:10]
    
    def getLinearAcceleration(self,frame='b'):
        """Returns the linear acceleration vector in either {b} or {n}. Default is {b}.

        :frame: Frame in which the linear acceleration vector is requested.
        :returns: acc_{frame}

        """
        acc = self.getForce()/self.model.Param['mass']
        if frame == 'b':
            return acc 
        elif frame == 'n':
            return self.getRotation_nb() @ acc
    
    def getAngularVelocity(self, frame='b'):
        """Returns the angular velocity vector in either {b} (default) or {n}.

        :frame: Requested frame {b} or {n}
        :returns: omega_{frame} [rad/s]

        """
        if frame =='b':
            return self.state[10:]
        elif frame =='s':
            return self.getRotation_sb()@self.state[10:]
        elif frame =='w':
            return self.getRotation_wb()@self.state[10:]
        elif frame == 'n':
            return self.getRotation_nb()@self.state[10:]

    def getGroundspeed(self):
        """Returns the groundspeed of the vehicle.
        :returns: V_g [m/s]

        """
        return np.linalg.norm(self.getLinearVelocity())

    def getCourseAngle(self):
        """Returns the course angle (angle between horizontal velocity direction and north).
        :returns: chi [rad]

        """
        return ng.courseAngle(self.getLinearVelocity(frame='n'))

    def getFlightPathAngle(self):
        """Returns the flight-path angle in [rad].
        :returns: gamma [rad]

        """
        return ng.flightPathAngle(self.getLinearVelocity(frame='n'))
        
    def getWindVelocity(self, frame='b'):
        """Returns the wind velocity vector wrt. {b} (default) or {n}.
        :returns: w_{frame}
        """
        if frame=='b':
            return self.getRotation_nb().T @ self.wind_n
        elif frame=='n':
            return self.wind_n

    def getRelativeVelocity(self, frame='b'):
        """Returns the (air-) relative velocity vector in {b} (default) or {n}.
        :returns: v_r in {frame}

        """
        return self.getLinearVelocity(frame) - self.getWindVelocity(frame) 

    def getAirspeed(self):
        """TODO: Docstring for getAirspeed.
        :returns: TODO

        """
        return np.linalg.norm(self.getRelativeVelocity()) 

    def getAOA(self) -> object:
        """TODO: Docstring for getAOA.
        :returns: TODO

        """
        return ng.aoa(self.getRelativeVelocity())

    def getSSA(self):
        """TODO: Docstring for getSSA.
        :returns: TODO

        """
        return ng.ssa(self.getRelativeVelocity()) 

    def getState(self):
        return self.state

    def getForce(self):
        """TODO: Docstring for getForce.
        :returns: TODO

        """
        return self.tau[0:3]

    def getThrustForce(self):
        pass
    
    def getDragForce(self):
        pass

    def getsideForce(self):
        pass

    def getLiftForce(self):
        pass

    def getMoment(self):
        """TODO: Docstring for getMoment.
        :returns: TODO

        """
        return self.tau[3:]

    def getRollMoment(self):
        pass

    def getPitchMoment(self):
        pass 

    def getYawMoment(self):
        pass

    def getControl(self):
        """TODO: Docstring for getInput.
        :returns: TODO

        """
        return np.array((self.aileron, self.elevator, self.rudder, self.throttle)).reshape(4)

    def getParam(self):
        return self.model.Param

    def getSensorValue(self,sensor):

        return self.sensorValue[sensor]

    def setAileronCmd(self, aileronCmd):
        """TODO: Docstring for setAileron.

        :aileron: TODO
        :returns: TODO

        """
        self.aileronCmd = aileronCmd
        return

    def setElevatorCmd(self, elevatorCmd):
        """TODO: Docstring for setElevator.

        :elevator: TODO
        :returns: TODO

        """
        self.elevatorCmd = elevatorCmd
        return

    def setRudderCmd(self, rudderCmd):
        """TODO: Docstring for setRudder.

        :rudder: TODO
        :returns: TODO

        """
        self.rudderCmd = rudderCmd
        return

    def setThrottleCmd(self, throttleCmd):
        """TODO: Docstring for setThrottle.

        :throttle: TODO
        :returns: TODO

        """
        self.throttleCmd = throttleCmd
        return

    def setWind(self, wind_n):
        """TODO: Docstring for setWind.

        :wind_n: TODO
        :returns: TODO

        """
        self.wind_n = wind_n
        return

    def isWingsLevelFlight(self):
        """TODO: Docstring for wingsLevelFlight.
        :returns: TODO

        """
        return (np.linalg.norm(self.getAngularVelocity()) < 1e-3 and self.getRollAngle() < 1e-3)
