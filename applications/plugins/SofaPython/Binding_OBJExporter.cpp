/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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


#include "Binding_OBJExporter.h"
#include "Binding_BaseObject.h"
#include "PythonToSofa.inl"

using namespace sofa::component::misc;
using namespace sofa::core::objectmodel;


/// getting a OBJExporter* from a PyObject*
static inline OBJExporter* get_OBJExporter(PyObject* obj) {
    return sofa::py::unwrap<OBJExporter>(obj);
}


static PyObject * OBJExporter_writeOBJ(PyObject *self, PyObject * /*args*/)
{
    OBJExporter* obj = get_OBJExporter( self );
    return PyBool_FromLong( obj->writeOBJ() ) ;
}


SP_CLASS_METHODS_BEGIN(OBJExporter)
SP_CLASS_METHOD(OBJExporter,writeOBJ)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(OBJExporter,OBJExporter,BaseObject)


