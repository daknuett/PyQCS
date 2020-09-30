import pytest
import ray
import numpy as np

from pyqcs.experiment.nelder_mead import nelder_mead
from pyqcs.experiment.workflow import WorkflowSpawner, FunctionInstruction

@pytest.fixture(scope="session")
def ray_setup():
    return ray.init()

def test_nelder_mead_convergence(ray_setup):
    f = lambda x: np.sum(x**2 + x**6)
    x0 = np.array(list(range(3)))
    instructions = [FunctionInstruction("quadratic function", f)]
    wf_spawner = WorkflowSpawner("quadratic function evaluation", instructions)

    success, (result, value) = nelder_mead(wf_spawner, x0, nmax=120_000, simplex_guess_parameter=6)

    assert success is True
    assert np.allclose(result, np.zeros_like(x0))
    assert np.allclose(value, 0)
