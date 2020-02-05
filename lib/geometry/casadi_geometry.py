import numpy as np
from sys import path
path.append(r"/home/dirkpr/casadi_all/casadi_py35")
import casadi as cs
import casadi as casadi
#from lib import numeric_geometry as ng

def Smtrx(x):
    """Returns a skew-symmetric matrix based on x."""
    S = cs.SX.zeros(3,3)
    S[0,0] = 0
    S[0,1] = -x[2]
    S[0,2] = x[1]
    S[1,0] = x[2]
    S[1,1] = 0
    S[1,2] = -x[0]
    S[2,0] = -x[1]
    S[2,1] = x[0]
    S[2,2] = 0

    return S

    # return np.array([[0, -x[2], x[1]],
    #                  [x[2], 0, -x[0]],
    #                  [-x[1], x[0], 0]])


def orthogonal_projector(x):
    """TODO: Docstring for orthogonal_projector.

    :x: TODO
    :returns: TODO

    """
    S = Smtrx(x)
    return -S@S

def parallel_projector(x):
    """TODO: Docstring for parallel_projector.

    :x: TODO
    :returns: TODO

    """
    return x@x.T

def quaternion_axis_angle(axis, angle):
    """Returns a quaternion from an axis, angle representation."""
    eta = np.cos(angle/2)
    epsilon = axis * np.sin(angle/2)
    return qNormalize(np.hstack((eta, epsilon)))


# Functions
def rotation_x(angle):
    """Returns a rotation matrix for a left-handed rotation around the x-axis by the given angle."""
    c, s = cs.cos(angle), cs.sin(angle)
    R = cs.SX.zeros(3,3)
    R[0,0], R[0,1], R[0,2] = 1,  0, 0
    R[1,0], R[1,1], R[1,2] = 0,  c, s
    R[2,0], R[2,1], R[2,2] = 0, -s, c
    return R

def rotation_y(angle):
    """Returns a rotation matrix for a left-handed rotation around the y-axis by the given angle."""
    c, s = cs.cos(angle), cs.sin(angle)
    R = cs.SX.zeros(3, 3)
    R[0,0],R[0,1],R[0,2] = c, 0, -s
    R[1,0],R[1,1],R[1,2] = 0, 1,  0
    R[2,0],R[2,1],R[2,2] = s, 0,  c
    return R

def rotation_z(angle):
    """Returns a rotation matrix for a left-handed rotation around the z-axis by the given angle."""
    c, s = cs.cos(angle), cs.sin(angle)
    R = cs.SX.zeros(3, 3)
    R[0,0],R[0,1],R[0,2] =  c, s, 0
    R[1,0],R[1,1],R[1,2] = -s, c, 0
    R[2,0],R[2,1],R[2,2] =  0, 0, 1
    return R

def rotation_rpy(roll, pitch, yaw):
    """Returns a rotation matrix from roll pitch yaw. ZYX convention."""
    cr = cs.cos(roll)
    sr = cs.sin(roll)
    cp = cs.cos(pitch)
    sp = cs.sin(pitch)
    cy = cs.cos(yaw)
    sy = cs.sin(yaw)

    R = cs.SX.zeros(3,3)
    R[0,0],R[0,1],R[0,2] = cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr
    R[1,0],R[1,1],R[1,2] = sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr
    R[2,0],R[2,1],R[2,2] = -sp,   cp*sr,            cp*cr

    return R

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
    return cs.arctan2(vel_r[2], vel_r[0])
    
def ssa(vel_r):
    """ Returns the sideslip angle."""
    return cs.arctan2(vel_r[1], vel_r[0])

def airspeed(vel_r):
    """ Returns the airspeed (magnitude of the relative velocity vector)."""
    return cs.norm_2(vel_r)

def quaternion_rpy(roll, pitch, yaw):
    """Returns a quaternion ([x,y,z,w], w scasadi.ar) from roll pitch yaw ZYX
    convention."""
    cr = cs.cos(roll/2.0)
    sr = cs.sin(roll/2.0)
    cp = cs.cos(pitch/2.0)
    sp = cs.sin(pitch/2.0)
    cy = cs.cos(yaw/2.0)
    sy = cs.sin(yaw/2.0)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    # Remember to normalize:
    nq = cs.sqrt(x*x + y*y + z*z + w*w)
    return cs.vertcat(w/nq, x/nq, y/nq, z/nq)

def quaternion_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = casadi.SX.zeros(4)
    w0, x0, y0, z0 = quat0[0], quat0[1], quat0[2], quat0[3]
    w1, x1, y1, z1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    quat[1] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[2] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[3] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return quat

def xi_quaternion(quat):
    return np.vstack((-quat[1:], quat[0]* np.identity(3) - Smtrx(quat[1:])))

def rotation_quaternion(quat):
    """Returns a rotation matrix from a quaternion."""
    #Rotation Matrix
    rowX_vb = casadi.horzcat(1-2*(quat[2]**2+quat[3]**2),2*(quat[1]*quat[2]-quat[3]*quat[0]),2*(quat[1]*quat[3]+quat[2]*quat[0]))
    rowY_vb = casadi.horzcat(2*(quat[1]*quat[2]+quat[3]*quat[0]),1-2*(quat[1]**2+quat[3]**2),2*(quat[2]*quat[3]-quat[1]*quat[0]))
    rowZ_vb = casadi.horzcat(2*(quat[1]*quat[3]-quat[2]*quat[0]),2*(quat[2]*quat[3]+quat[1]*quat[0]),1-2*(quat[1]**2+quat[2]**2))
    return casadi.vertcat(rowX_vb, \
                       rowY_vb, \
                       rowZ_vb)

def rotation_axis_angle(axis, ang):
    """Returns a rotation matrix from an axis, angle representation."""
    return np.identity(3) - np.sin(ang) * Smtrx(axis) + (1 - np.cos(ang)) * np.linalg.matrix_power(Smtrx(axis), 2)
    
#def rotation_matrix_quaternion(quat):
#    return np.identity(3) + 2 * quat[0] * Smtrx(quat[1:]) + 2 * np.linalg.matrix_power(Smtrx(quat[1:]), 2)

# From Markley book for Attitude matrix
#def rotation_matrix_quaternion(q):
#    return (q[0] ** 2 - norm(q[1:]) ** 2) * np.identity(3) - 2 * q[0] * Smtrx(q[1:]) + 2 * np.outer(q[1:], q[1:])

def rpy_quaternion(quat):
    """ Returns roll, pitch, yaw angle (ZYX-Convention) from a quaternion."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    roll = cs.arctan2(2.0*(qw*qx + qy*qz), 1 - 2*(qx**2+qy**2))
    pitch = cs.arcsin(2*(qw*qy-qz*qx))
    yaw = cs.arctan2(2*(qw*qz+qx*qy), 1 - 2*(qy**2 + qz**2))
    return roll, pitch, yaw
    

#def quaternion_dot(quat, angVel):
#    """ Returns the time-derivative of a quaternion from the quaternion and angular velocity."""
#    #return 0.5 * np.dot(psi_quaternion(quat), angVel)
#    # Equation 3: angular rates = rotational matrix * angular velocity
#    #quat_dot = 0.5*vert.([-quat[1:3].T,quat[0]*SX.eye(2)])*vert.([0,q,0])
#    return 0.5*casadi.vertcat(casadi.horzcat(-quat[1],-quat[2],-quat[3]),\
#                                         casadi.horzcat(quat[0],-quat[3],quat[2]),\
#                                         casadi.horzcat(quat[3],quat[0],-quat[1]),\
#                                         casadi.horzcat(-quat[2],quat[1],quat[0])) @ angVel 
    
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

def quaternion_dot(quat, angVel):
    """ Returns the time-derivative of a quaternion from the quaternion and angular velocity."""
    return 0.5 * psi_quaternion(quat)@angVel

def psi_quaternion(quat):
    return cs.vertcat(-quat[1:].T, quat[0] * cs.SX.eye(3) + Smtrx(quat[1:]))
    
def courseAngle(vel_n):
    """ Returns the course angle."""
    return cs.arctan2(vel_n[1], vel_n[0])

def flightPathAngle(vel_n):
    """ Returns the flight-path angle."""
    return cs.arcsin(-vel_n[2]/cs.sqrt(vel_n.T@vel_n))#cs.linalg.norm(vel_n))   

