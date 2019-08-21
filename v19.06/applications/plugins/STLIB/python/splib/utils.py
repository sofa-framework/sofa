# -*- coding: utf-8 -*-
"""
Utilitary function & decorator

**********
Functions:
**********

.. autosummary::

"""
import Sofa
import functools

def deprecated_alias(**aliases):
    """Decorator to wrap old parameters name to new one
    
    Example:
        suppose the function def animate(cb): is now def animate(onUpdate):
        you can decorate it so it still handle the old names for backward
        compatibility in the following way:

        @deprecated_alias(cb='onUpdate')
        def animate(onUpdate):
    """
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            __rename_kwargs__(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)
        return wrapper
    return deco

def __rename_kwargs__(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError("'{}' received both '{}' and '{}' while '{}' is deprecated".format(
                    func_name, alias, new, alias))
            Sofa.msg_deprecated("{}' is a deprecated parameter name which has been replaced with {}.".format(alias, new))
            kwargs[new] = kwargs.pop(alias)

