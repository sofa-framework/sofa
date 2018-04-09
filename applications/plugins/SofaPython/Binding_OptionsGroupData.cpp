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
#include "Binding_OptionsGroupData.h"
#include "Binding_Data.h"
#include "PythonToSofa.inl"


using namespace sofa::helper;
using namespace sofa::core::objectmodel;





/// getting a Data<OptionsGroup>* from a PyObject*
static inline Data<OptionsGroup>* get_DataOptionsGroup(PyObject* obj) {
    return sofa::py::unwrap<Data<OptionsGroup> >(obj);
}

static PyObject * OptionsGroupData_getAttr_selectedItem(PyObject *self, void*)
{
    Data<OptionsGroup>* data = get_DataOptionsGroup( self );
    return PyString_FromString(data->getValue().getSelectedItem().c_str());
}

static int OptionsGroupData_setAttr_selectedItem_impl(PyObject *self, char* item)
{
    Data<OptionsGroup>* data = get_DataOptionsGroup( self );
    OptionsGroup* optionGroups = data->beginEdit();
    optionGroups->setSelectedItem( item );
    data->endEdit();
    return 0;
}

static int OptionsGroupData_setAttr_selectedItem(PyObject *self, PyObject * args, void*)
{
    char *str = PyString_AsString(args); // for setters, only one object and not a tuple....
    OptionsGroupData_setAttr_selectedItem_impl(self,str);
    return 0;
}

static PyObject * OptionsGroupData_getAttr_selectedId(PyObject *self, void*)
{
    Data<OptionsGroup>* data = get_DataOptionsGroup( self );
    return PyInt_FromLong((long)data->getValue().getSelectedId());
}

void OptionsGroupData_setAttr_selectedId_impl(PyObject *self, unsigned id)
{
    Data<OptionsGroup>* data = get_DataOptionsGroup( self );
    OptionsGroup* optionGroups = data->beginEdit();
    optionGroups->setSelectedItem( (unsigned)id );
    data->endEdit();
}

static int OptionsGroupData_setAttr_selectedId(PyObject *self, PyObject * args, void*)
{
    OptionsGroupData_setAttr_selectedId_impl( self, (unsigned)PyInt_AsLong(args) );
    return 0;
}


static PyObject * OptionsGroupData_getSelectedId(PyObject *self, PyObject *)
{
    return OptionsGroupData_getAttr_selectedId(self,NULL);
}

static PyObject * OptionsGroupData_setSelectedId(PyObject *self, PyObject * args)
{
    int index;
    if (!PyArg_ParseTuple(args, "i",&index))
    {
        return NULL;
    }
    OptionsGroupData_setAttr_selectedId_impl(self,index);
    Py_RETURN_NONE;
}

static PyObject * OptionsGroupData_getSelectedItem(PyObject *self, PyObject *)
{
    return OptionsGroupData_getAttr_selectedItem(self,NULL);
}

static PyObject * OptionsGroupData_setSelectedItem(PyObject *self, PyObject * args)
{
    char *item;
    if (!PyArg_ParseTuple(args, "s",&item))
    {
        return NULL;
    }
    OptionsGroupData_setAttr_selectedItem_impl(self,item);
    Py_RETURN_NONE;
}

static PyObject * OptionsGroupData_getItem(PyObject *self, PyObject * args)
{
    Data<OptionsGroup>* data = get_DataOptionsGroup( self );
    int index;
    if (!PyArg_ParseTuple(args, "i",&index))
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyString_FromString(data->getValue()[index].c_str());
}

static PyObject * OptionsGroupData_getSize(PyObject *self, PyObject *)
{
    Data<OptionsGroup>* data = get_DataOptionsGroup( self );
    return PyInt_FromLong((long)data->getValue().size());
}



SP_CLASS_ATTRS_BEGIN(OptionsGroupData)
SP_CLASS_ATTR(OptionsGroupData,selectedItem)
SP_CLASS_ATTR(OptionsGroupData,selectedId)
SP_CLASS_ATTRS_END



SP_CLASS_METHODS_BEGIN(OptionsGroupData)
SP_CLASS_METHOD(OptionsGroupData,getSize)
SP_CLASS_METHOD(OptionsGroupData,getItem)
SP_CLASS_METHOD(OptionsGroupData,getSelectedItem)
SP_CLASS_METHOD(OptionsGroupData,setSelectedItem)
SP_CLASS_METHOD(OptionsGroupData,getSelectedId)
SP_CLASS_METHOD(OptionsGroupData,setSelectedId)
SP_CLASS_METHODS_END



SP_CLASS_TYPE_PTR_ATTR(OptionsGroupData, BaseData, Data);

