#" File: pid.py
#" Author: dirk
#" Description: Generic class for a discrete PID controller
#" Last Modified: mars 04, 2019

import numpy as np

class PID(object):

    """Docstring for PID. """

    def __init__(self, kp, ki, kd, I_min, I_max, fc, u_min=-np.inf, u_max=np.inf):
        """TODO: to be defined1.

        :kp: TODO
        :ki: TODO
        :kd: TODO
        :D: TODO
        :I: TODO
        :I_min: TODO
        :I_max: TODO
        :fc: TODO

        """
        self._P = 0
        self._I = 0
        self._D = 0
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._I_min = I_min
        self._I_max = I_max
        self._fc = fc
        self._error = 0.0
        self._ref = 0.0
        self._RC = 1/(2*np.pi*fc)
        self._lastDerivative = np.nan
        self._lastError = np.nan
        self._last_t = 0
        self._u_min = u_min
        self._u_max = u_max

    def update(self, t,  error):
        """TODO: Docstring for update.

        :currentValue: TODO
        :returns: TODO

        """
        dt = t - self._last_t
        if dt == 0 or dt > 1e3:
            self.resetI()
            
        self._last_t = t

        self._error = error

        # Proportional term
        self._P = self._kp * self._error
        
        # Integral term
        if self._ki and dt > 0:
            self._I += self._ki*self._error*dt

            # Anti-windup
            if self._I < self._I_min:
                self._I = self._I_min
            elif self._I > self._I_max:
                self._I = self._I_max


        # Derivative term
        if self._kd and dt > 0:
            if np.isnan(self._lastDerivative):
                derivative = 0
                self._lastDerivative = 0
            else:
                derivative = (error - self._lastError)/dt

            # Apply low-pass filter
            derivative = self._lastDerivative + \
                    (dt/(self._RC + dt))*(derivative - self._lastDerivative)

            self._lastError = error
            self._lastDerivative = derivative

            self._D = self._kd*derivative

        u = np.clip(sum((self._P, self._I, self._D)), self._u_min, self._u_max) 
        return u

    def setRef(self, ref):
        self._ref = ref

    def setI(self, I):
        self._I = I

    def setD(self, D):
        self._D = D

    def resetI(self):
        self._I = 0

    def resetD(self):
        self._D = 0
        
    def setKp(self, kp):
        self._kp = kp

    def setKi(self, ki):
        self._ki = ki

    def setKd(self, kd):
        self._kd = kd

    def getP(self):
        return self._P

    def getI(self):
        return self._I

    def getD(self):
        return self._D
