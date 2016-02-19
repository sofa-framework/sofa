# -*- coding: UTF8 -*-
from cython.operator cimport dereference as deref, preincrement as inc, address as address
include "vec3d.pyx"
include "base.pyx"
include "baseobjectdescription.pyx"
include "objectfactory.pyx"
include "basedata.pyx"
include "baseobject.pyx"
include "basenode.pyx"
include "basecontext.pyx"
include "node.pyx"
include "simulation.pyx"

# Let's include now the different components.
include "basemechanicalstate.pyx"
