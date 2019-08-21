# -*- coding: utf-8 -*-
"""
Easing function to use in animation

.. autosummary::

   LinearRamp

splib.animation.easing.LinearRamp
*********************************
.. autofunction:: LinearRamp

"""

import math

def LinearRamp(beginValue=0.0, endValue=1.0, scale=0.5):
    """Linear interpolation between two values

    Examples:
        LinearRamp(10, 20, 0.5)    

    """
    return (endValue-beginValue) * scale + beginValue


