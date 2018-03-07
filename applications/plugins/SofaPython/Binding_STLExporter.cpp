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


#include "Binding_STLExporter.h"
#include "Binding_BaseObject.h"
#include "PythonToSofa.inl"

using namespace sofa::component::misc;
using namespace sofa::core::objectmodel;

/// getting a STLExporter* from a PyObject*
static inline STLExporter* get_STLExporter(PyObject* obj) {
    return sofa::py::unwrap<STLExporter>(obj);
}

static PyObject * STLExporter_writeSTL(PyObject *self, PyObject * /*args*/)
{
    STLExporter* obj = get_STLExporter( self );
    obj->writeSTL();
    Py_RETURN_NONE;
}

static PyObject * STLExporter_writeSTLBinary(PyObject *self, PyObject * /*args*/)
{
    STLExporter* obj = get_STLExporter( self );
    obj->writeSTLBinary();
    Py_RETURN_NONE;
}


SP_CLASS_METHODS_BEGIN(STLExporter)
SP_CLASS_METHOD(STLExporter,writeSTL)
SP_CLASS_METHOD(STLExporter,writeSTLBinary)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(STLExporter,STLExporter,BaseObject)


