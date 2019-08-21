# -*- coding: utf-8 -*-
"""
Templates to ease collision and contact handling.

**Content:**

.. autosummary::

   CollisionMesh

|

stlib.physics.collision.CollisionMesh
*************************************
.. autofunction:: CollisionMesh

"""

from collision import CollisionMesh


def FrictionalContact(applyTo=None):
    applyTo.createObject('CollisionResponse', response="FrictionContact", responseParams="mu=0")
    applyTo.createObject('LocalMinDistance', name="Proximity", alarmDistance="3", contactDistance="1")
