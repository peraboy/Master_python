import numpy as np
import matplotlib.pyplot as plt
from lib.geometry import numpy_geometry as ng
import matplotlib as mpl
import numpy as np


def get_color():
    rgb_red   = (0.7725,0.21560,0.1098) # red
    rgb_green = (0.0000,0.6588, 0.3647) # gree
    rgb_blue  = (0.0000,0.3647, 0.6588) # blue
    return [rgb_red, rgb_green, rgb_blue]

def set_options(kwargs):
    if 'line_style' not in kwargs:
        line_style = '-'
    else:
        line_style = kwargs['line_style']


    if 'color' not in kwargs:
        # Set the default color cycle
        color = get_color()
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[color[0], color[1], color[2]])

    opt = dict()
    opt['line_style'] = line_style
    opt['color'] = color
    return opt
    # opt['color'] = color
    # opt['line_style'] = line_style
    # return line_style

    

def plotState(t, X, **kwargs):
    """TODO: Docstring for plotState.

    :t: TODO
    :pos_n: TODO
    :rpy_nb: TODO
    :linVel_b: TODO
    :angVel_b: TODO
    :returns: TODO

    """
    X = X.T

    if 'attitudeRepresentation' not in kwargs:
        attitudeRepresentation = 'euler'
    else:
        attitudeRepresentation = kwargs['attitudeRepresentation']

    if attitudeRepresentation == 'euler':
        attitude = np.zeros((len(t), 3))
        for i in range(0, len(t)):
            attitude[i, 0], attitude[i, 1], attitude[i, 2] = ng.rpy_quaternion(X[i, 3:7])
        attitude = attitude * 180.0/np.pi 
        attitudeLegend = ('roll', 'pitch', 'yaw')
        attitudeLabel = 'Angle [deg]'
    elif attitudeRepresentation == 'quaternion':
        attitude = X[:, 3:7]
        attitudeLegend = ('qw', 'qx', 'qy', 'qz')
        attitudeLabel = 'q_i [-]'


    if 'axes' not in kwargs:
        fig, axes = plt.subplots(2, 2) 
    else:
        fig = kwargs['fig']
        axes = kwargs['axes']

    for i in range(0, axes.shape[0]):
        for j in range(0, axes.shape[1]):
            axes[i, j].set_color_cycle(None)

    opt = set_options(kwargs)
    line_style = opt['line_style']
    axes[0, 0].plot(t, X[:, 0:3], linestyle = line_style)
    axes[0, 0].legend(('N', 'E', 'D'))
    axes[0, 0].set_ylabel('Position [m]')

    axes[0, 1].plot(t, attitude, linestyle = line_style)
    axes[0, 1].legend(attitudeLegend)
    axes[0, 1].set_ylabel(attitudeLabel)

    axes[1, 0].plot(t, X[:, 7:10], linestyle = line_style)
    axes[1, 0].legend(('u', 'v', 'w'))
    axes[1, 0].set_ylabel('Linear velocity [m/s]')

    axes[1, 1].plot(t, X[:, 10:] * 180/np.pi, linestyle = line_style)
    axes[1, 1].set_ylabel('Angular rate [deg/s]')
    axes[1, 1].legend(('p', 'q', 'r'))

    for i in range(0, axes.shape[0]):
        for j in range(0, axes.shape[1]):
            axes[i, j].set_xlabel('Time [s]')
            axes[i, j].grid(True)

    return fig, axes

def plotInput(t, U, **kwargs):
    """TODO: Docstring for plotInput.

    :t: TODO
    :U: TODO
    :returns: TODO

    """
    U = U.T

    if 'axes' not in kwargs:
        fig, axes = plt.subplots() 
    else:
        fig = kwargs['fig']
        axes = kwargs['axes']

    opt = set_options(kwargs)
    line_style = opt['line_style']

    axes.plot(t, U, linestyle = line_style)
    axes.legend(('aileron', 'elevator', 'rudder', 'throttle'))
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Input [-]')
    axes.grid(True)
    return fig, axes 

def plotForce(t, force, moment, **kwargs):
    """TODO: Docstring for plotForce.

    :t: TODO
    :F: TODO
    :M: TODO
    :returns: TODO

    """
    force = force.T
    moment = moment.T


    if 'axes' not in kwargs:
        fig, axes = plt.subplots(2, 1) 
    else:
        fig = kwargs['fig']
        axes = kwargs['axes']

    opt = set_options(kwargs)
    line_style = opt['line_style']

    axes[0].plot(t, force, linestyle = line_style)
    axes[0].legend(('X', 'Y', 'Z'))
    axes[0].set_ylabel('Force [N]')

    axes[1].plot(t, moment, linestyle = line_style)
    axes[1].legend(('roll', 'pitch', 'yaw'))
    axes[1].set_ylabel('Moment [N/m]')
    
    for iAxes in range(0, axes.size):
        axes[iAxes].set_xlabel('Time [s]')
        axes[iAxes].grid(True)

    return fig, axes

def plotRelativeVelocity(t, airspeed, aoa, ssa, **kwargs):
    """TODO: Docstring for plotRelativeVelocity.

    :t: TODO
    :airspeed: TODO
    :aoa: TODO
    :ssa: TODO
    :returns: TODO

    """

    if 'axes' not in kwargs:
        fig, axes = plt.subplots(3, 1) 
    else:
        fig = kwargs['fig']
        axes = kwargs['axes']

    opt = set_options(kwargs)
    line_style = opt['line_style']

    axes[0].plot(t, airspeed, linestyle = line_style)
    axes[0].set_ylabel('Airspeed [m/s]')

    axes[1].plot(t, aoa*180/np.pi, linestyle = line_style)
    axes[1].set_ylabel('AOA [deg]')

    axes[2].plot(t, ssa*180/np.pi, linestyle = line_style)
    axes[2].set_ylabel('SSA [deg]')

    for iAxes in range(0, axes.size):
        axes[iAxes].set_xlabel('Time [s]')
        axes[iAxes].grid(True)


    return fig, axes
    
