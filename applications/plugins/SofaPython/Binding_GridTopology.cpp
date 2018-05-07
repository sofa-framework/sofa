/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "Binding_GridTopology.h"
#include "Binding_MeshTopology.h"
#include "PythonToSofa.inl"

#include <SofaBaseTopology/GridTopology.h>
using sofa::component::topology::GridTopology ;

/// getting a GridTopology* from a PyObject*
static inline GridTopology* get_GridTopology(PyObject* obj) {
    return sofa::py::unwrap<GridTopology>(obj);
}


static PyObject * GridTopology_setSize(PyObject *self, PyObject * args)
{
    GridTopology* obj = get_GridTopology( self );
    int nx,ny,nz;
    if (!PyArg_ParseTuple(args, "iii",&nx,&ny,&nz))
    {
        return NULL;
    }
    obj->setSize(nx,ny,nz);
    Py_RETURN_NONE;
}


static PyObject * GridTopology_getNx(PyObject *self, PyObject * /*args*/)
{
    GridTopology* obj = get_GridTopology( self );
    return PyInt_FromLong(obj->getNx());
}

static PyObject * GridTopology_setNx(PyObject *self, PyObject * args)
{
    GridTopology* obj = get_GridTopology( self );
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        return NULL;
    }
    obj->setNx(nb);
    Py_RETURN_NONE;
}

static PyObject * GridTopology_getNy(PyObject *self, PyObject * /*args*/)
{
    GridTopology* obj = get_GridTopology( self );
    return PyInt_FromLong(obj->getNy());
}

static PyObject * GridTopology_setNy(PyObject *self, PyObject * args)
{
    GridTopology* obj = get_GridTopology( self );
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        return NULL;
    }
    obj->setNy(nb);
    Py_RETURN_NONE;
}

static PyObject * GridTopology_getNz(PyObject *self, PyObject * /*args*/)
{
    GridTopology* obj = get_GridTopology( self );
    return PyInt_FromLong(obj->getNz());
}

static PyObject * GridTopology_setNz(PyObject *self, PyObject * args)
{
    GridTopology* obj = get_GridTopology( self );
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        return NULL;
    }
    obj->setNz(nb);
    Py_RETURN_NONE;
}


SP_CLASS_METHODS_BEGIN(GridTopology)
SP_CLASS_METHOD(GridTopology,setSize)
SP_CLASS_METHOD(GridTopology,getNx)
SP_CLASS_METHOD(GridTopology,getNy)
SP_CLASS_METHOD(GridTopology,getNz)
SP_CLASS_METHOD(GridTopology,setNx)
SP_CLASS_METHOD(GridTopology,setNy)
SP_CLASS_METHOD(GridTopology,setNz)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(GridTopology,GridTopology,MeshTopology)




