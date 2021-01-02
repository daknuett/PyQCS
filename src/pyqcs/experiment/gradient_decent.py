from collections import deque
import numpy as np

try:
    import ray
except ImportError:
    raise ImportError("pyqcs experiments require ray for parallelization; install it using pip3 install ray")

def approximate_gradient(pool, x, eps):
    """
    Compute an appoximate gradient using

    .. math::

        F'(x) \\approx \\frac{F(x + \\epsilon) - F(x - \\epsilon)}{2\\epsilon}

    ``pool`` is an ``ActorPool`` that evaluates the objective
    function.
    """
    evaluation_points = deque()
    for i,_ in enumerate(x):
        ei = np.zeros_like(x, dtype=np.double)
        ei[i] = eps
        evaluation_points.append(x + ei)
        evaluation_points.append(x - ei)

    data_points = list(pool.map(lambda a,v: a.execute.remote(v), evaluation_points))

    gradient = np.array([(data_points[2*i] - data_points[2*i + 1])/(2*eps) for i,_ in enumerate(x)])
    return gradient

def approx_gradient_decent(wf_spawner, x0, eps, gamma, nmax=40_000, rho=1e-6):
    """
    Perform a gradient decent using an appoximate gradient (computed
    by ``approximate_gradient``; ``rho`` is passed to approximate_gradient as
    parameter ``eps``).

    ``wf_spawner`` is a ``WorkflowSpawner`` used to compute the objective function.

    ``x0`` is the starting point, ``eps`` is the minimal norm of the gradient to
    continue, ``gamma`` is the step width parameter, and ``nmax`` specifies how
    many steps are made at most.

    Returns ``success: bool, x_final``.

    Uses an ``ActorPool`` with ``len(x0)`` actors.

    This function expects the objective function to be costly.
    Therefore all function evaluations are parallelized.
    """
    pool = ray.util.ActorPool([wf_spawner.spawn() for _ in x0])
    x = x0

    for n in range(nmax):
        gradient = approximate_gradient(pool, x, rho)
        if(np.linalg.norm(gradient) < eps):
            return True, x

        x = x - gamma*gradient

    if(np.linalg.norm(gradient) < eps):
        return True, x
    return False, x

