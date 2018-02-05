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

/// @author M Nesme @date 2016
///
///
/// TODO: we could add a [] operator to get bound DataFileName

#include "Binding_DataFileNameVector.h"
#include "Binding_Data.h"
#include "PythonToSofa.inl"

using sofa::core::objectmodel::DataFileNameVector ;

/// getting a DataFileNameVector* from a PyObject*
static inline DataFileNameVector* get_DataFileNameVector(PyObject* obj) {
    return sofa::py::unwrap<DataFileNameVector>(obj);
}


static PyObject * DataFileNameVector_clear(PyObject *self, PyObject *)
{
    DataFileNameVector* data = get_DataFileNameVector( self );

    sofa::helper::vector<std::string>& val = *data->beginEdit();
    val.clear();
    data->endEdit();

    Py_RETURN_NONE;
}

static PyObject * DataFileNameVector_addPath(PyObject *self, PyObject *args)
{
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
        return nullptr;

    DataFileNameVector* data = get_DataFileNameVector( self );

    data->addPath(path);

    Py_RETURN_NONE;
}


SP_CLASS_ATTRS_BEGIN(DataFileNameVector)
SP_CLASS_ATTRS_END

SP_CLASS_METHODS_BEGIN(DataFileNameVector)
SP_CLASS_METHOD(DataFileNameVector,addPath)
SP_CLASS_METHOD(DataFileNameVector,clear)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_PTR_ATTR(DataFileNameVector, sofa::core::objectmodel::BaseData, Data);

