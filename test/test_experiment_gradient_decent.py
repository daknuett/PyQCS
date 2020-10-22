import pytest
import ray
import numpy as np

from pyqcs.experiment.gradient_decent import approx_gradient_decent
from pyqcs.experiment.workflow import WorkflowSpawner, FunctionInstruction

@pytest.mark.slow
def test_gradient_decent_convergence(ray_setup):
    f = lambda x: np.sum(x**2 + x**6)
    x0 = np.array(list(range(3)))
    instructions = [FunctionInstruction("quadratic function", f)]
    wf_spawner = WorkflowSpawner("quadratic function evaluation", instructions)

    success, result = approx_gradient_decent(wf_spawner, x0, 1e-5, 2e-3)

    assert success is True
    assert np.allclose(result, np.zeros_like(x0), atol=1e-5)
