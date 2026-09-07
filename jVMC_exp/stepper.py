from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp

def _rms_norm(v):
    """ 
    Root-mean-square norm :math:`\\|v\\|_2/\\sqrt{N}`.

    Default norm of the adaptive steppers. Unlike the Euclidean norm it does not grow
    with the number of degrees of freedom, so ``atol`` and ``rtol`` retain their meaning
    as per-parameter tolerances irrespective of the size of the ansatz.
    """
    return jnp.linalg.norm(v) / jnp.sqrt(jnp.size(v))

def _clip(x, c_min, c_max):
    return max(min(x, c_max), c_min)

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

class AbstractStepper(ABC):
    """ 
    Base class of all steppers.

    Initializer arguments:
        * ``time_step``: Time step :math:`\\Delta t`. Has to be positive.
    """
    def __init__(self, time_step: float):
        if time_step <= 0:
            raise ValueError(
                f'The value of time_step has to be positive. Got {time_step}'
            )
        self.dt = time_step
        super().__init__()

    @abstractmethod
    def step(self, t, f, yInitial, **kwargs):
        pass

class AbstractAdaptiveStepper(AbstractStepper):
    """ 
    Base class of the adaptive steppers.

    Holds the step size controller shared by :class:`AdaptiveHeun`, :class:`RK23` and
    :class:`RK45`. Derived classes only supply the Butcher tableau and the pair of
    approximations that is compared.

    Initializer arguments:
        * ``error_order``: Order :math:`p` of the error estimate, i.e. the order of \
                the *lower* order member of the pair. For an embedded pair this is one \
                less than the order of the propagated solution.
        * ``time_step``: Initial time step (will be adapted automatically). Clipped to \
                ``[min_step, max_step]``.
        * ``rtol``: Relative tolerance, weighting :math:`\\|y\\|`.
        * ``atol``: Absolute tolerance.
        * ``min_step``: Smallest allowed time step, or ``None`` for no lower bound. \
                When the controller asks for a smaller step the current one is accepted \
                regardless of its error and :math:`\\Delta t` is held at ``min_step``.
        * ``max_step``: Largest allowed time step, or ``None`` for no upper bound.
    """
    def __init__(
            self, error_order, time_step, rtol, atol, min_step: float | None, max_step: float | None
        ):
        self.max_step = max_step
        self.min_step = min_step
        if self.min_step >= self.max_step:
            raise ValueError(
                f"min_step ({self.min_step}) must be smaller than max_step ({self.max_step})"
            )
        self._order = error_order
        self.rtol = rtol
        self.atol = atol

        super().__init__(_clip(time_step, self.min_step, self.max_step))

    @property
    def order(self):
        """ Order :math:`p` of the error estimate, entering the controller as :math:`\\epsilon^{-1/(p+1)}`.

        This is the order of the error estimate, *not* of the propagated solution.
        """
        return self._order

    @property
    def min_step(self):
        return self._min_step

    @min_step.setter
    def min_step(self, value):
        self._min_step = -1 if value is None else value

    @property
    def max_step(self):
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        self._max_step = 1e10 if value is None else value

    def _adaptive_step_control(self, y_norm, dy_low, dy_high, norm_function):
        """ 
        Accept or reject the current step and propose the next time step.

        Arguments:
            * ``y_norm``: Norm of the current value of :math:`y`, measured with ``norm_function``.
            * ``dy_low``: Update obtained from the lower order member of the pair.
            * ``dy_high``: Update obtained from the higher order member of the pair.
            * ``norm_function``: Norm used to quantify the magnitude of the error.

        Returns:
            Whether the step is accepted, and the time step :math:`\\Delta t` to use next.
        """
        update_diff = norm_function(dy_low - dy_high)
        if not jnp.isfinite(update_diff):
            raise RuntimeError(
                f"Adaptive step control got a non-finite error estimate ({update_diff}). "
                "The right-hand side of the ODE is NaN or inf -- check the solver"
            )

        scaled_diff = update_diff / (self.atol + y_norm * self.rtol)
        exponent = -1 / (self.order + 1)
        dt_rescale = _clip(
            scaled_diff**exponent * 0.95,
            c_min=0.2,
            c_max=2
        )
        dt_new = self.dt * dt_rescale

        converged = scaled_diff < 1 or dt_new < self.min_step

        return converged, _clip(dt_new, self.min_step, self.max_step)

class Euler(AbstractStepper):
    ''' 
    This class implements Euler integration
    '''
    def __init__(self, timeStep: float | Callable=1e-3):
        self._scheduler = None
        if isinstance(timeStep, Callable):
            self._scheduler = timeStep
            dt = timeStep(0)
        else:
            dt = timeStep
        super().__init__(dt)

    def update_dt(self, step: int):
        if self._scheduler is not None:
            self.dt = self._scheduler(step)

    def step(self, t, f, yInitial, **rhsArgs):
        """ 
        Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes the Euler integration step

        :math:`y_{n+1} = y_n+\\Delta t f(y_n,t_n,p)`

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
        dy = f(yInitial, t, **rhsArgs, intStep=0)

        return yInitial + self.dt * dy, self.dt

class Heun(AbstractStepper):
    """ 
    Implement a second order consistent integration scheme with fixed time step.

    See :class:`AdaptiveHeun` for the variant that adapts the time step automatically.

    Initializer arguments:
        * ``timeStep``: Time step.
    """
    def __init__(self, timeStep=1e-3):
        super().__init__(timeStep)

        self._butcher_tableau = (
                jnp.array([[0, 0], [1, 0]]),
                jnp.array([0.5, 0.5]),
                jnp.array([0, 1])
            )

    def step(self, t, f, yInitial, **rhsArgs):
        """ 
        Perform an integration time step.

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

class AdaptiveHeun(AbstractAdaptiveStepper):
    """ 
    Implement an adaptive second order consistent integration scheme.

    The error is estimated by step doubling: one Heun step of size :math:`\\Delta t` is
    compared with two Heun steps of size :math:`\\Delta t/2`. Both are second order
    accurate, so their difference scales as :math:`\\Delta t^3` and the order of the error
    estimate is 2. The more accurate half-step solution is the one that is propagated.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically).
        * ``rtol``: Relative tolerance for integration errors.
        * ``atol``: Absolute tolerance for integration errors.
        * ``min_step``: Minimal allowed time step, or ``None`` for no lower bound.
        * ``max_step``: Maximal allowed time step, or ``None`` for no upper bound.
    """
    def __init__(self, timeStep=1e-3, rtol=1e-5, atol=1e-8, min_step=None, max_step=1):
        super().__init__(2, timeStep, rtol, atol, min_step, max_step)

        self._butcher_tableau = (
            jnp.array([[0, 0], [1, 0]]),
            jnp.array([0.5, 0.5]),
            jnp.array([0, 1])
        )

    def step(self, t, f, y, *, normFunction=_rms_norm, **rhsArgs):
        converged = False
        k0 = f(y, t, **rhsArgs, intStep=0)
        y_norm = normFunction(y)

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
            converged, self.dt = self._adaptive_step_control(y_norm, dy0, dy1, normFunction)

        return y + dy1, current_dt
    
class RK23(AbstractAdaptiveStepper):
    """
    This class implements the explicit Runge-Kutta method of order 3(2) by Bogacki and Shampine.

    The third order solution is propagated and the embedded second order solution supplies
    the error estimate, so the order of the error estimate is 2. The scheme is FSAL -- the
    extra stage evaluated for the error estimate equals the first stage of the next step --
    but this is not exploited yet, see the ``TODO`` in :meth:`step`.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically).
        * ``rtol``: Relative tolerance for integration errors.
        * ``atol``: Absolute tolerance for integration errors.
        * ``min_step``: Minimal allowed time step, or ``None`` for no lower bound.
        * ``max_step``: Maximal allowed time step, or ``None`` for no upper bound.
    """
    def __init__(self, timeStep=1e-3, rtol=1e-5, atol=1e-8, min_step=None, max_step=1):
        super().__init__(2, timeStep, rtol, atol, min_step, max_step)
        self._k0 = None

        self._butcher_tableau = (
            jnp.array([[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]]),
            jnp.array([2/9, 1/3, 4/9]),
            jnp.array([0, 0.5, 0.75])
        )

    def step(self, t, f, y, *, normFunction=_rms_norm, **rhsArgs):
        """ Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a third-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors. \
                    Defaults to the root-mean-square norm, see the module documentation.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        converged = False
        # k0 = self._k0 if self._k0 is not None else f(y, t, **rhsArgs, intStep=0)
        # TODO: at the moment this is needed to trigger intStep=0, but the above line saves a step
        k0 = f(y, t, **rhsArgs, intStep=0)
        y_norm = normFunction(y)

        while not converged:
            dy_high, K = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            y_new = y + dy_high
            K.append(f(y_new, t + self.dt , **rhsArgs, intStep=3))
            dy_low = self.dt * (7/24 * K[0] + 1/4 * K[1] + 1/3 * K[2] + 1/8 * K[3])

            current_dt = self.dt
            converged, self.dt = self._adaptive_step_control(y_norm, dy_low, dy_high, normFunction)
            self._k0 = K[-1] if converged else None

        return y_new, current_dt
    
class RK45(AbstractAdaptiveStepper):
    """
    This class implements the explicit Runge-Kutta method of order 5(4) by Dormand and Prince.

    The fifth order solution is propagated and the embedded fourth order solution supplies
    the error estimate, so the order of the error estimate is 4. The scheme is FSAL -- the
    extra stage evaluated for the error estimate equals the first stage of the next step --
    but this is not exploited yet, see the ``TODO`` in :meth:`step`.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically).
        * ``rtol``: Relative tolerance for integration errors.
        * ``atol``: Absolute tolerance for integration errors.
        * ``min_step``: Minimal allowed time step, or ``None`` for no lower bound.
        * ``max_step``: Maximal allowed time step, or ``None`` for no upper bound.
    """
    def __init__(self, timeStep=1e-3, rtol=1e-5, atol=1e-8, min_step=None, max_step=1):
        super().__init__(4, timeStep, rtol, atol, min_step, max_step)
        self._k0 = None

        self._butcher_tableau = (
            jnp.array([
                [0, 0, 0, 0, 0, 0], 
                [0.2, 0, 0, 0, 0, 0], 
                [3/40, 9/40, 0, 0, 0, 0],
                [44/45, -56/15, 32/9, 0, 0, 0],
                [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
                [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
            ]),
            jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]),
            jnp.array([0, 0.2, 3/10, 0.8, 8/9, 1])
        )

    def step(self, t, f, y, *, normFunction=_rms_norm, **rhsArgs):
        """ Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a fifth-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors. \
                    Defaults to the root-mean-square norm, see the module documentation.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        converged = False
        # k0 = self._k0 if self._k0 is not None else f(y, t, **rhsArgs, intStep=0)
        # TODO: at the moment this is needed to trigger intStep=0, but the above line saves a step
        k0 = f(y, t, **rhsArgs, intStep=0)
        y_norm = normFunction(y)

        while not converged:
            dy_high, K = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            y_new = y + dy_high
            K.append(f(y_new, t + self.dt , **rhsArgs, intStep=6))
            dy_low = self.dt * (5179/57600 * K[0] + 7571/16695 * K[2] + 393/640 * K[3] - 92097/339200 * K[4] + 187/2100 * K[5] + 1/40 * K[6])

            current_dt = self.dt
            converged, self.dt = self._adaptive_step_control(y_norm, dy_low, dy_high, normFunction)
            self._k0 = K[-1] if converged else None

        return y_new, current_dt