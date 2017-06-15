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

#include "Binding_GridTopology.h"
#include "Binding_MeshTopology.h"

using namespace sofa::component::topology;
using namespace sofa::core::objectmodel;

extern "C" PyObject * GridTopology_setSize(PyObject *self, PyObject * args)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    int nx,ny,nz;
    if (!PyArg_ParseTuple(args, "iii",&nx,&ny,&nz))
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->setSize(nx,ny,nz);
    Py_RETURN_NONE;
}

extern "C" PyObject * GridTopology_getNx(PyObject *self, PyObject * /*args*/)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    return PyInt_FromLong(obj->getNx());
}

extern "C" PyObject * GridTopology_setNx(PyObject *self, PyObject * args)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->setNx(nb);
    Py_RETURN_NONE;
}

extern "C" PyObject * GridTopology_getNy(PyObject *self, PyObject * /*args*/)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    return PyInt_FromLong(obj->getNy());
}

extern "C" PyObject * GridTopology_setNy(PyObject *self, PyObject * args)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->setNy(nb);
    Py_RETURN_NONE;
}

extern "C" PyObject * GridTopology_getNz(PyObject *self, PyObject * /*args*/)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    return PyInt_FromLong(obj->getNz());
}

extern "C" PyObject * GridTopology_setNz(PyObject *self, PyObject * args)
{
    GridTopology* obj=down_cast<GridTopology>(((PySPtr<Base>*)self)->object->toTopology());
    int nb;
    if (!PyArg_ParseTuple(args, "i",&nb))
    {
        PyErr_BadArgument();
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




