class Actuator(object):

    """Docstring for Actuator. """

    def __init__(self, state, ref, T, fs, lb, ub):
        """Implements a first-order actuator

        :state: TODO
        :ref: TODO
        :T: TODO
        :fs: TODO
        :lb: TODO
        :ub: TODO

        """
        self._state = state
        self._ref = ref
        self._T = T
        self._fs = fs
        self._lb = lb
        self._ub = ub
    def update(self):
        """Updates the actuator stateue according to x_dot = 1/T*(ref - x)
        """
        self._state = self._state + (1/self._fs)*(self._ref - self._state)/self._T
        self._state = max(self._state, self._lb)
        self._state = min(self._state, self._ub)

    def get_state(self):
        return self._state

    def set_reference(self, ref):
        self._ref = ref
