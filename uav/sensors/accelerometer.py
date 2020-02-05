import numpy as np
from .. import uav

class Accelerometer(object):

    """Docstring for Accelerometer. """

    def __init__(self,std=[0,0,0],bias=[0,0,0]):
        """TODO: to be defined1. """
        self.bias = np.vstack(bias)
        if len(std) == 1:
            self.std = std
        else:
            self.std = np.vstack(std)

    def update(self, uav):
        return uav.getLinearAcceleration() + self.bias + self.std*np.random.randn(3,1) - uav.getRotation_nb().T@np.vstack([0,0,uav.model.Param['gravity']])
