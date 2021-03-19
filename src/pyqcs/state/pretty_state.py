from .state import BasicState


class PrettyState(BasicState):
    """
    This state inherits from ``BasicState``. Refer to its documentation
    for more information.

    The ``PrettyState`` class is a drop-in replacement for ``basic_state``
    that essentially just replaces the ``__str__`` method with something
    more pretty.
    """

    def _easy_format_state_part(self, cf, i):
        nbits = str(self._nbits)
        return ("({0.real: .3e} {0.imag:+.3e}j)*|{1:0=" + nbits + "b}>").format(cf, i)

    def __str__(self):
        eps = 1e-13

        s = "   " + "\n + ".join((self._easy_format_state_part(self._qm_state[i], i)
                        for i in range(self._ndim) if (abs(self._qm_state[i]) > eps)))
        return s
