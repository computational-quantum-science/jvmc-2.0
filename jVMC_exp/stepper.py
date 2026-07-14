import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Callable

def _get_rk_step(butcher_tableau, t, f, y0, dt, k0=None, start_step=0, **rhs_kwargs):
    '''
    The Butcher tableau of the explicit Runge-Kutta scheme has to be given as a tuple
    (A, b, c).

    The tableau defines the integration scheme through:

        k_i = f(
            y_n + dt * sum_j A[i,j] * k_j,
            t_n + c[i] * dt
        )

    and the final update:

        y_{n+1} = y_n + dt * sum_i b[i] * k_i

    where:
        * A : strictly lower-triangular matrix of stage coefficients
        * b : vector of coefficients for the final weighted sum
        * c : vector of time shifts for each stage

    The tableau must follow the standard explicit RK convention:
        * len(A) == len(b) == len(c)
        * c[0] == 0
        * A[0,:] == 0
        * A[i,j] = 0 for j >= i
    '''
    A, b, c = butcher_tableau

    K = [f(y0, t, **rhs_kwargs, intStep=start_step)] if k0 is None else [k0]
    step = b[0] * K[0]

    for i in range(1, len(b)):
        y_stage = y0 + dt * sum(A[i][j] * K[j] for j in range(i))
        t_stage = t + c[i] * dt

        K.append(f(y_stage, t_stage, **rhs_kwargs, intStep=i + start_step))
        step += b[i] * K[i]

    return dt * step, K

def _adaptive_step_control(dy_low, dy_high, tolerance, max_step, norm_function, dt, order):
    update_diff = norm_function(dy_low - dy_high)
    fe = tolerance / update_diff

    if 0.2 > 0.9 * fe**(1 / (order + 1)):
        tmp = 0.2
    else:
        tmp = 0.9 * fe**(1 / (order + 1))
    dt_new = dt * min(2, tmp)

    return fe > 1., min(dt_new, max_step)

class AbstractStepper(ABC):
    @abstractmethod
    def step(self, t, f, yInitial, **kwargs):
        pass

class Euler(AbstractStepper):
    ''' 
    This class implements Euler integration
    '''

    def __init__(self, timeStep: float | Callable=1e-3):
        self._scheduler = None
        if isinstance(timeStep, Callable):
            self._scheduler = timeStep
            self.dt = timeStep(0)
        else:
            self.dt = timeStep

    def update_dt(self, step: int):
        if self._scheduler is not None:
            self.dt = self._scheduler(step)

    def step(self, t, f, yInitial, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes the Euler integration step

        :math:`y_{n+1} = y_n+\\Delta t f(y_n,t_n,p)`

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """

        dy = f(yInitial, t, **rhsArgs, intStep=0)

        return yInitial + self.dt * dy, self.dt

class Heun(AbstractStepper):
    """ This class implements an adaptive second order consistent integration scheme.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
    """

    def __init__(self, timeStep=1e-3):
        self.dt = timeStep

        self._butcher_tableau = (
                jnp.array([[0, 0], [1, 0]]),
                jnp.array([0.5, 0.5]),
                jnp.array([0, 1])
            )

    def step(self, t, f, yInitial, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``yInitial``: Initial value of :math:`y`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """

        dy0, _ = _get_rk_step(
            self._butcher_tableau, 
            t, f, yInitial, self.dt, **rhsArgs
        )

        return yInitial + dy0, self.dt

class AdaptiveHeun(AbstractStepper):
    """ This class implements an adaptive second order consistent integration scheme.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
        * ``tol``: Tolerance for integration errors.
        * ``maxStep``: Maximal allowed time step.
    """

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep

        self._butcher_tableau = (
            jnp.array([[0, 0], [1, 0]]),
            jnp.array([0.5, 0.5]),
            jnp.array([0, 1])
        )

    def step(
        self, t, f, y,
        *, normFunction=jnp.linalg.norm, **rhsArgs
    ):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        converged = False

        k0 = f(y, t, **rhsArgs, intStep=0) # TODO: this was inside the loop before
        
        while not converged:    
            dy0, _ = _get_rk_step(
                self._butcher_tableau, 
                t, f, y, self.dt, k0, **rhsArgs
            )
            dy1, _ = _get_rk_step(
                self._butcher_tableau, 
                t, f, y, 0.5 * self.dt, k0, 1, **rhsArgs
            )
            dy1_half, _ = _get_rk_step(
                self._butcher_tableau, 
                t + 0.5 * self.dt, f, y + dy1, 0.5 * self.dt, start_step=3, **rhsArgs
            )
            dy1 += dy1_half

            current_dt = self.dt
            converged, self.dt = _adaptive_step_control(dy0, dy1, self.tolerance, self.maxStep, normFunction, self.dt, order=2)

        return y + dy1, current_dt
    
class RK23(AbstractStepper):
    """ 
    This class implements the explicit Runge-Kutta method of order 2(3) by Bogacki and Shampine.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
        * ``tol``: Tolerance for integration errors.
        * ``maxStep``: Maximal allowed time step.
    """

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep
        self._k0 = None

        self._butcher_tableau = (
            jnp.array([[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]]),
            jnp.array([2/9, 1/3, 4/9]),
            jnp.array([0, 0.5, 0.75])
        )

    def step(self, t, f, y, normFunction=jnp.linalg.norm, **rhsArgs):
        converged = False
        # k0 = self._k0 if self._k0 is not None else f(y, t, **rhsArgs, intStep=0) 
        # TODO: at the moment this is needed to trigger intStep=0, but the above line saves a step
        k0 = f(y, t, **rhsArgs, intStep=0)

        while not converged:
            dy_high, K = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            y_new = y + dy_high
            K.append(f(y_new, t + self.dt , **rhsArgs, intStep=3))
            dy_low = self.dt * (7/24 * K[0] + 1/4 * K[1] + 1/3 * K[2] + 1/8 * K[3])

            current_dt = self.dt
            converged, self.dt = _adaptive_step_control(dy_low, dy_high, self.tolerance, self.maxStep, normFunction, self.dt, order=3)
            self._k0 = K[-1] if converged else None

        return y_new, current_dt
    
class RK45(AbstractStepper):
    """ 
    This class implements the explicit Runge-Kutta method of order 4(5) by Dormand and Prince.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
        * ``tol``: Tolerance for integration errors.
        * ``maxStep``: Maximal allowed time step.
    """

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep
        self._k0 = None

        self._butcher_tableau = (
            jnp.array([
                [0, 0, 0, 0, 0, 0], 
                [0.2, 0, 0, 0, 0, 0], 
                [3/40, 9/40, 0, 0, 0, 0],
                [44/45, -56/15, 32/9, 0, 0, 0],
                [19375/6561, -25460/2187, 64448/6561, -212/729, 0, 0],
                [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
            ]),
            jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]),
            jnp.array([0, 0.2, 3/10, 0.8, 8/9, 1])
        )

    def step(self, t, f, y, normFunction=jnp.linalg.norm, **rhsArgs):
        converged = False
        # k0 = self._k0 if self._k0 is not None else f(y, t, **rhsArgs, intStep=0) 
        # TODO: at the moment this is needed to trigger intStep=0, but the above line saves a step
        k0 = f(y, t, **rhsArgs, intStep=0)

        while not converged:
            dy_high, K = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            y_new = y + dy_high
            K.append(f(y_new, t + self.dt , **rhsArgs, intStep=6))
            dy_low = self.dt * (5179/57600 * K[0] + 7571/16695 * K[2] + 393/640 * K[3] - 92097/339200 * K[4] + 187/2100 * K[5] + 1/40 * K[6])

            current_dt = self.dt
            converged, self.dt = _adaptive_step_control(dy_low, dy_high, self.tolerance, self.maxStep, normFunction, self.dt, order=5)
            self._k0 = K[-1] if converged else None

        return y_new, current_dt