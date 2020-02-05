import numpy as np
#from lib import numeric_geometry as ng

def Smtrx(x):
    """Returns a skew-symmetric matrix based on x."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def quaternion_axis_angle(axis, angle):
    """Returns a quaternion from an axis, angle representation."""
    eta = np.cos(angle/2)
    epsilon = axis * np.sin(angle/2)
    return qNormalize(np.hstack((eta, epsilon)))


def rotation_rpy(roll, pitch, yaw):
    """Returns a rotation matrix from roll pitch yaw. ZYX convention."""
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                     [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                     [  -sp,             cp*sr,             cp*cr]])


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rpy_rotation(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def quaternion_rpy(roll, pitch, yaw):
    """Returns a quaternion ([x,y,z,w], w scalar) from roll pitch yaw ZYX
    convention."""
    cr = np.cos(roll/2.0)
    sr = np.sin(roll/2.0)
    cp = np.cos(pitch/2.0)
    sp = np.sin(pitch/2.0)
    cy = np.cos(yaw/2.0)
    sy = np.sin(yaw/2.0)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    # Remember to normalize:
    nq = np.sqrt(x*x + y*y + z*z + w*w)
    return np.array([w/nq,
                     x/nq,
                     y/nq,
                     z/nq])

def quaternion_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = np.zeros(4)
    w0, x0, y0, z0 = quat0[0], quat0[1], quat0[2], quat0[3]
    w1, x1, y1, z1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    quat[1] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[2] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[3] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return quat

def quaternion_rotation(R):
    # See Fos)sen p. 32
    R_33 = np.trace(R)
    temp = np.array((R[0,0], R[1,1], R[2,2], R_33))
    i = temp.argmax()
    p_i = np.sqrt(1+2*temp[i] - R_33)

    if i==0:
       p0 = p_i
       p1 = (R[1,0]+R[0,1])/p_i
       p2 = (R[0,2]+R[2,0])/p_i
       p3 = (R[2,1]-R[1,2])/p_i
    elif i==1:
       p0 = (R[1,0]+R[0,1])/p_i
       p1 = p_i
       p2 = (R[2,1]+R[1,2])/p_i
       p3 = (R[0,2]-R[2,0])/p_i
    elif i==2:
       p0 = (R[0,2]+R[2,0])/p_i
       p1 = (R[2,1]+R[1,2])/p_i
       p2 = p_i
       p3 = (R[1,0]-R[0,1])/p_i
    else:
       p0 = (R[2,1]-R[1,2])/p_i
       p1 = (R[0,2]-R[2,0])/p_i
       p2 = (R[1,0]-R[0,1])/p_i
       p3 = p_i

    q = 0.5* np.array([p3, p0, p1, p2])
    q = q/(np.dot(q, q))

    return q

def xi_quaternion(quat):
    return np.vstack((-quat[1:].T, quat[0]* np.identity(3) - Smtrx(quat[1:])))

def psi_quaternion(quat):
    return np.vstack((-quat[1:].T, quat[0] * np.identity(3) + Smtrx(quat[1:])))

def rotation_quaternion(quat):
    """Returns a rotation matrix from a quaternion."""
    return xi_quaternion(quat).transpose() @ psi_quaternion(quat)

def rotation_quaternion_2(quat):
    R = np.zeros((3,3))
    R[0,0] = 1 - 2*(quat[2]**2 + quat[3]**2)
    R[0,1] = 2*(quat[1]*quat[2] - quat[3]*quat[0])
    R[0,2] = 2*(quat[1]*quat[3] + quat[2]*quat[0])
    R[1,0] = 2*(quat[1]*quat[2] + quat[3]*quat[0])
    R[1,1] = 1 - 2*(quat[1]**2 + quat[3]**2)
    R[1,2] = 2*(quat[2]*quat[3] - quat[1]*quat[0])
    R[2,0] = 2*(quat[1]*quat[3] - quat[2]*quat[0])
    R[2,1] = 2*(quat[2]*quat[3] + quat[1]*quat[0])
    R[2,2] = 1 - 2*(quat[1]**2 + quat[2]**2)
    return R

def rotation_quaternion_3(quat):
    pass

def rotation_axis_angle(axis, ang):
    """Returns a rotation matrix from an axis, angle representation."""
    return np.identity(3) - np.sin(ang) * Smtrx(axis) + (1 - np.cos(ang)) * np.linalg.matrix_power(Smtrx(axis), 2)
    
#def rotation_matrix_quaternion(quat):
#    return np.identity(3) + 2 * quat[0] * Smtrx(quat[1:]) + 2 * np.linalg.matrix_power(Smtrx(quat[1:]), 2)

# From Markley book for Attitude matrix
#def rotation_matrix_quaternion(q):
#    return (q[0] ** 2 - norm(q[1:]) ** 2) * np.identity(3) - 2 * q[0] * Smtrx(q[1:]) + 2 * np.outer(q[1:], q[1:])

def rollAngle_quaternion(quat):
    """TODO: Docstring for rollAngle_quaternion.

    :quat: TODO
    :returns: TODO

    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return np.arctan2(2.0*(qw*qx + qy*qz), 1 - 2*(qx**2+qy**2))

def pitchAngle_quaternion(quat):
    """TODO: Docstring for pitchAngle_quaternion.

    :quat: TODO
    :returns: TODO

    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return np.arcsin(2*(qw*qy-qz*qx))

def yawAngle_quaternion(quat):
    """TODO: Docstring for yawAngle_quaternion.

    :quat: TODO
    :returns: TODO

    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return np.arctan2(2*(qw*qz+qx*qy), 1 - 2*(qy**2 + qz**2))

def rpy_quaternion(quat):
    """ Returns roll, pitch, yaw angle (ZYX-Convention) from a quaternion."""
    roll = rollAngle_quaternion(quat)
    pitch = pitchAngle_quaternion(quat)
    yaw = yawAngle_quaternion(quat)
    return roll, pitch, yaw
    

def quaternion_dot(quat, angVel):
    """ Returns the time-derivative of a quaternion from the quaternion and angular velocity."""
    return 0.5 * psi_quaternion(quat)@angVel
    
#def quaternion_product(quat0, quat1):
#    return quat0 * quat1[0] + quaternion_psi(quat0) @ quat1[1:]
    
def quaternion_vector_transformation(quat, v):
    """Transforms the vector v from frame A to frame B given the quaternion representing the orientation of frame B relative to frame A."""
    return quaternion_xi(quat).transpose() @ quaternion_psi(quat) @ v

def quatNormalize(quat):
    """Returns a normlized quaternion."""
    return quat/np.linalg.norm(quat)

def quatIdentity():
    """Returns the quaternion-identity [1,0,0,0]."""
    return np.array((1, 0, 0, 0))

def quatConj(quat):
    """Returns the quaternion-conjuagte."""
    return quat * np.array((1, -1, -1, -1))

def quatInv(quat):
    """Returns the quaternion inverse."""
    return quatConj(quat) / np.linalg.norm(quat)
    
def rotation_x(angle):
    """Returns a rotation matrix for a left-handed rotation around the x-axis by the given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.vstack([[1, 0, 0], [0, c, s], [0, -s, c]])

def rotation_y(angle):
    """Returns a rotation matrix for a left-handed rotation around the y-axis by the given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.vstack([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def rotation_z(angle):
    """Returns a rotation matrix for a left-handed rotation around the z-axis by the given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.vstack([[c, s, 0], [-s, c, 0], [0, 0, 1]])

def rotation_sb(alpha):
    """ Returns a rotation matrix from body frame to stability frame by alpha. """
    return rotation_y(-alpha)

def rotation_ws(beta):
    """ Returns a rotation matrix from stability frame to wind frame by beta. """
    return rotation_z(beta)

def rotation_wb(alpha, beta):
    """ Returns a rotation matrix from body frame to wind frame by alpha, beta. """
    return rotation_ws(beta)@rotation_sb(alpha)

def aoa(vel_r):
    """ Returns the angle of attack."""
    return np.asscalar(np.arctan2(vel_r[2], vel_r[0]))
    
def ssa(vel_r):
    """ Returns the side-slip angle."""
    return np.arctan2(vel_r[1], vel_r[0])
    # return np.asscalar(np.arcsin(vel_r[1]/np.linalg.norm(vel_r)))

def airspeed(vel_r):
    """ Returns the airspeed (magnitude of the relative velocity vector)."""
    return np.linalg.norm(vel_r)

def courseAngle(vel_n):
    """ Returns the course angle."""
    return np.arctan2(vel_n[1], vel_n[0])

def flightPathAngle(vel_n):
    """ Returns the flight-path angle."""
    return np.arcsin(-vel_n[2]/np.linalg.norm(vel_n))   


