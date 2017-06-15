/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/


#include "Binding_Topology.h"
#include "Binding_BaseObject.h"

using namespace sofa::core::topology;
using namespace sofa::core;
using namespace sofa::core::objectmodel;

extern "C" PyObject * Topology_hasPos(PyObject *self, PyObject * /*args*/)
{
    Topology* obj=((PySPtr<Base>*)self)->object->toTopology();
    return PyBool_FromLong(obj->hasPos());
}

extern "C" PyObject * Topology_getNbPoints(PyObject *self, PyObject * /*args*/)
{
    Topology* obj=((PySPtr<Base>*)self)->object->toTopology();
    return PyInt_FromLong(obj->getNbPoints());
}

extern "C" PyObject * Topology_setNbPoints(PyObject *self, PyObject * args)
{
    Topology* obj=((PySPtr<Base>*)self)->object->toTopology();
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->setNbPoints(nb);
    Py_RETURN_NONE;
}

extern "C" PyObject * Topology_getPX(PyObject *self, PyObject * args)
{
    Topology* obj=((PySPtr<Base>*)self)->object->toTopology();
    int i;
    if (!PyArg_ParseTuple(args, "i",&i))
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyFloat_FromDouble(obj->getPX(i));
}

extern "C" PyObject * Topology_getPY(PyObject *self, PyObject * args)
{
    Topology* obj=((PySPtr<Base>*)self)->object->toTopology();
    int i;
    if (!PyArg_ParseTuple(args, "i",&i))
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyFloat_FromDouble(obj->getPY(i));
}

extern "C" PyObject * Topology_getPZ(PyObject *self, PyObject * args)
{
    Topology* obj=((PySPtr<Base>*)self)->object->toTopology();
    int i;
    if (!PyArg_ParseTuple(args, "i",&i))
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyFloat_FromDouble(obj->getPZ(i));
}



SP_CLASS_METHODS_BEGIN(Topology)
SP_CLASS_METHOD(Topology,hasPos)
SP_CLASS_METHOD(Topology,getNbPoints)
SP_CLASS_METHOD(Topology,setNbPoints)
SP_CLASS_METHOD(Topology,getPX)
SP_CLASS_METHOD(Topology,getPY)
SP_CLASS_METHOD(Topology,getPZ)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(Topology,Topology,BaseObject)



