"""
This package provides noise channels for PyQCS.
Noise is emulated using noisy executors. An executor
can be noisified using the ``noisify`` function. 

"""

from .executor import noisify
