import pytest

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

def pytest_collection_modifyitems(config, items):
    if(config.getoption("--runslow")
            and config.getoption("--rundeprecated")):
        return

    skip_slow = pytest.mark.skip(reason="slow test")
    skip_deprecated = pytest.mark.skip(reason="deprecated test")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "deprecated" in item.keywords:
            item.add_marker(skip_deprecated)


