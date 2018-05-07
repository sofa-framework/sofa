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

#include "Binding_BaseMeshTopology.h"
#include "Binding_Topology.h"
#include "PythonToSofa.inl"

using namespace sofa::core::topology;
using namespace sofa::core::objectmodel;

static BaseMeshTopology* get_basemeshtopology(PyObject* self) {
    return sofa::py::unwrap<BaseMeshTopology>(self);
}


static PyObject * BaseMeshTopology_getNbEdges(PyObject *self, PyObject * /*args*/)
{
    BaseMeshTopology* obj = get_basemeshtopology( self );
    return PyInt_FromLong(obj->getNbEdges());
}

static PyObject * BaseMeshTopology_getNbTriangles(PyObject *self, PyObject * /*args*/)
{
    BaseMeshTopology* obj = get_basemeshtopology( self );
    return PyInt_FromLong(obj->getNbTriangles());
}

static PyObject * BaseMeshTopology_getNbQuads(PyObject *self, PyObject * /*args*/)
{
    BaseMeshTopology* obj = get_basemeshtopology( self );
    return PyInt_FromLong(obj->getNbQuads());
}

static PyObject * BaseMeshTopology_getNbTetrahedra(PyObject *self, PyObject * /*args*/)
{
    BaseMeshTopology* obj = get_basemeshtopology( self );
    return PyInt_FromLong(obj->getNbTetrahedra());
}

static PyObject * BaseMeshTopology_getNbHexahedra(PyObject *self, PyObject * /*args*/)
{
    BaseMeshTopology* obj = get_basemeshtopology( self );
    return PyInt_FromLong(obj->getNbHexahedra());
}


SP_CLASS_METHODS_BEGIN(BaseMeshTopology)
SP_CLASS_METHOD(BaseMeshTopology,getNbEdges)
SP_CLASS_METHOD(BaseMeshTopology,getNbTriangles)
SP_CLASS_METHOD(BaseMeshTopology,getNbQuads)
SP_CLASS_METHOD(BaseMeshTopology,getNbTetrahedra)
SP_CLASS_METHOD(BaseMeshTopology,getNbHexahedra)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(BaseMeshTopology,BaseMeshTopology,Topology)



