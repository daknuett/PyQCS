import numpy as np

try:
    import ray
except ImportError:
    raise ImportError("pyqcs experiments require ray for parallelization; install it using pip3 install ray")

def build_initial_simplex(x0, simplex_guess_parameter=None):
    """
    Build an initial simplex from x0.
    Algorithm from https://stackoverflow.com/a/19282873.

    If ``simplex_guess_parameter is None`` the ``hi`` are chosen
    as mentioned in the answer above; else ``hi = simplex_guess_parameter``.
    """
    x0 = np.array(x0)
    simplex = [x0]
    for i,x in enumerate(x0):
        ei = np.zeros_like(x0)
        ei[i] = 1

        if(simplex_guess_parameter is None):
            if(np.allclose(x, 0)):
                hi = 0.00025
            else:
                hi = 0.05
        else:
            hi = simplex_guess_parameter

        simplex.append(x0 + ei*hi)

    return np.array(simplex)




def nelder_mead(wf_spawner, x0
                , nmax=1000, eps=1e-4
                , simplex=None
                , parallelize_aggressively=False
                , alpha=lambda N: 1
                , beta=lambda N: 1 + 2/N
                , gamma=lambda N: 0.75 - 1/(2*N)
                , delta=lambda N: 1 - 1/N
                , simplex_guess_parameter=None
                ):
    """
    Uses the Nelder-Mead Algorithm to minimize the function computed using
    ``wf_spawner.spawn().execute.remote(x)``.

    The method is taken from
    *F. Gao, L. Han, "Implementing the Nelder-Mead simplex algorithm with adaptive parameters", Comput. Optim. Appl., DOI 10.1007/s10589-010-9329-3*
    available at http://www.webpages.uidaho.edu/~fuchang/res/ANMS.pdf.
    The parameters are functions of ``N = len(x0)``, default values are given by the paper.

    If you want to disable this feature use::

        alpha=lambda N: 1
        beta=lambda N: 2
        gamma=lambda N: 0.5
        delta=lambda N: 0.5

    Note that the algorithm given in the paper mentioned above is not comprehensive.
    *Lagarias, J.C., Reeds, J.A., Wright, M.H., Wright, P.: Convergence properties of the Nelder-Meadsimplex algorithm in low dimensions. SIAM J. Optim.9, 112–147 (1998)*
    give the same algorithm with more explanation.


    If ``simplex is None`` an initial simplex is computed using ``build_initial_simplex``;
    ``simplex_guess_parameter`` is passed through and can be used to adjust the size of the
    simplex.

    Uses at most ``nmax`` iterations and is terminated if ``variance([f(i) for i in simplex]) < eps``.

    Returns ``success: bool, min((x,f(x)) for x in simplex, key=lambda x: x[1])``.

    This function expects the function that is supposed to be minimized to be costly
    (typically a quantum simulation with sampling). It therefore tries to
    avoid evaluating the minimizable. If the parameter ``parallelize_aggressively is True``
    (set it to true, iff there is one single optimization running) it computes
    all possibly needed evaluations in parallel.


    Uses ``max(len(simplex), 4)`` workers.
    """
    if(simplex is None):
        simplex = build_initial_simplex(x0)

    workers = [wf_spawner.spawn() for _ in range(max(len(simplex), 4))]
    pool = ray.util.ActorPool(workers)

    skip_evaluation = False
    N = len(x0)

    for n in range(1, nmax + 1):
        # Step 1
        if(not skip_evaluation):
            simplex_values = np.array(list(pool.map(lambda a,v: a.execute.remote(v), simplex)))

        if(np.var(simplex_values) < eps):
            return True, min(zip(simplex, simplex_values), key=lambda x: x[1])

        simplex = np.array([x for x,v in sorted(zip(simplex, simplex_values), key=lambda x: x[1])])
        simplex_values = np.sort(simplex_values)
        # End of step 1.

        centroid = np.average(simplex, axis=0)
        reflection = centroid + alpha(N)*(centroid - simplex[-1])
        expansion = centroid + beta(N)*(reflection - centroid)
        outside_contraction = centroid + gamma(N)*(reflection - centroid)
        inside_contraction = centroid - gamma(N)*(reflection - centroid)

        if(parallelize_aggressively):
            results = list(pool.map(lambda a,v: a.execute.remote(v)
                                 , [reflection, expansion, outside_contraction, inside_contraction]))
        else:
            results = []

        skip_inside_contraction = False

        # Step 2
        if(not parallelize_aggressively):
            results.extend(pool.map(lambda a,v: a.execute.remote(v), [reflection]))
        if(simplex_values[0] <= results[0] < simplex_values[-2]):
            simplex[-1,:] = reflection
            simplex_values[-1] = results[0]
            skip_evaluation = True
            continue
        # Step 3
        if(not parallelize_aggressively):
            results.extend(pool.map(lambda a,v: a.execute.remote(v), [expansion]))
        if(results[0] < simplex_values[0]):
            if(results[1] < results[0]):
                simplex[-1,:] = expansion
                simplex_values[-1] = results[1]
                skip_evaluation = True
                continue
            else:
                simplex[-1,:] = reflection
                simplex_values[-1] = results[0]
                skip_evaluation = True
                continue
        # Step 4
        if(not parallelize_aggressively):
            results.extend(pool.map(lambda a,v: a.execute.remote(v), [outside_contraction]))
        if(simplex_values[-2] <= results[0] < simplex_values[-1]):
            if(results[2] <= results[0]):
                simplex[-1,:] = outside_contraction
                simplex_values[-1] = results[2]
                skip_evaluation = True
                continue
            else:
                skip_inside_contraction = True

        # Step 5
        if(not parallelize_aggressively):
            results.extend(pool.map(lambda a,v: a.execute.remote(v), [inside_contraction]))
        if(not skip_inside_contraction and
                (results[0] >= simplex_values[-1])):
            if(results[3] < simplex_values[-1]):
                simplex[-1,:] = inside_contraction
                simplex_values[-1] = results[3]
                skip_evaluation = True
                continue

        # Step 6
        x1 = simplex[0]
        for i,xi in enumerate(simplex[1:]):
            xi = x1 + delta(n)*(xi - x1)
            simplex[i,:] = xi
        skip_evaluation = False

    if(np.var(simplex_values) < eps):
        return True, min(zip(simplex, simplex_values), key=lambda x: x[1])

    return False, min(zip(simplex, simplex_values), key=lambda x: x[1])

