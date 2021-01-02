import pytest
import ray

def pytest_addoption(parser):
    parser.addoption(
            "--runslow"
            , action="store_true"
            , default=False
            , help="run slow tests"
            )
    parser.addoption(
            "--rundeprecated"
            , action="store_true"
            , default=False
            , help="run deprecated tests"
            )
    parser.addoption(
            "--onlyselected"
            , action="store_true"
            , default=False
            , help="run only selected tests"
            )


def pytest_collection_modifyitems(config, items):
    if(config.getoption("--runslow")
            and config.getoption("--rundeprecated")
            and config.getoption("--onlyselected")):
        return

    if(not config.getoption("--onlyselected")):
        skip_slow = pytest.mark.skip(reason="slow test")
        skip_deprecated = pytest.mark.skip(reason="deprecated test")
        run_slow = config.getoption("--runslow")
        run_deprecated = config.getoption("--rundeprecated")
        for item in items:
            if not run_slow:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)
            if not run_deprecated:
                if "deprecated" in item.keywords:
                    item.add_marker(skip_deprecated)
    else:
        skip_not_selected = pytest.mark.skip(reason="not selected")
        for item in items:
            if not "selected" in item.keywords:
                item.add_marker(skip_not_selected)


@pytest.fixture(scope="session")
def ray_setup():
    return ray.init()
