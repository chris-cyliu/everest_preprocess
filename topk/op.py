"""Custom operator."""


class CustomOp(object):
    """Base class for operators."""
    def __init__(self):
        pass

    def forward(self, in_data, out_data):
        """Forward interface. Override to create new operators.

        Parameters
        ----------
        in_data, out_data: list
            input and output for forward.
        """
        pass
