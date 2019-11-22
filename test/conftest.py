import pytest

def pytest_addoption(parser):
    parser.addoption(
            "--runslow"
            , action="store_true"
            , default=False
            , help="run slow tests"
            )

def pytest_collection_modifyitems(config, items):
    if(config.getoption("--runslow")):
        return

    skip = pytest.mark.skip(reason="slow test")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


