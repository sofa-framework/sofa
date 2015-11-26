/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "Binding_OptionsGroupData.h"
#include "Binding_Data.h"

#include <sofa/helper/OptionsGroup.h>
using namespace sofa::helper;
using namespace sofa::core::objectmodel;




extern "C" PyObject * OptionsGroupData_getAttr_currentItem(PyObject *self, void*)
{
    Data<OptionsGroup>* data = down_cast<Data<OptionsGroup> >( ((PyPtr<BaseData>*)self)->object );
    return PyString_FromString(data->getValue().getSelectedItem().c_str());
}
extern "C" int OptionsGroupData_setAttr_currentItem_impl(PyObject *self, char* item)
{
    Data<OptionsGroup>* data = down_cast<Data<OptionsGroup> >( ((PyPtr<BaseData>*)self)->object );
    OptionsGroup* optionGroups = data->beginEdit();
    optionGroups->setSelectedItem( item );
    data->endEdit();
    return 0;
}
extern "C" int OptionsGroupData_setAttr_currentItem(PyObject *self, PyObject * args, void*)
{
    char *str = PyString_AsString(args); // for setters, only one object and not a tuple....
    OptionsGroupData_setAttr_currentItem_impl(self,str);
    return 0;
}

extern "C" PyObject * OptionsGroupData_getAttr_currentId(PyObject *self, void*)
{
    Data<OptionsGroup>* data = down_cast<Data<OptionsGroup> >( ((PyPtr<BaseData>*)self)->object );
    return PyInt_FromLong((long)data->getValue().getSelectedId());
}
void OptionsGroupData_setAttr_currentId_impl(PyObject *self, unsigned id)
{
    Data<OptionsGroup>* data = down_cast<Data<OptionsGroup> >( ((PyPtr<BaseData>*)self)->object );
    OptionsGroup* optionGroups = data->beginEdit();
    optionGroups->setSelectedItem( (unsigned)id );
    data->endEdit();
}
extern "C" int OptionsGroupData_setAttr_currentId(PyObject *self, PyObject * args, void*)
{
    OptionsGroupData_setAttr_currentId_impl( self, (unsigned)PyInt_AsLong(args) );
    return 0;
}


extern "C" PyObject * OptionsGroupData_getCurrentId(PyObject *self, PyObject *)
{
    return OptionsGroupData_getAttr_currentId(self,NULL);
}
extern "C" PyObject * OptionsGroupData_setCurrentId(PyObject *self, PyObject * args)
{
    int index;
    if (!PyArg_ParseTuple(args, "i",&index))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    OptionsGroupData_setAttr_currentId_impl(self,index);
    Py_RETURN_NONE;
}
extern "C" PyObject * OptionsGroupData_getCurrentItem(PyObject *self, PyObject *)
{
    return OptionsGroupData_getAttr_currentItem(self,NULL);
}
extern "C" PyObject * OptionsGroupData_setCurrentItem(PyObject *self, PyObject * args)
{
    char *item;
    if (!PyArg_ParseTuple(args, "s",&item))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    OptionsGroupData_setAttr_currentItem_impl(self,item);
    Py_RETURN_NONE;
}


extern "C" PyObject * OptionsGroupData_getItem(PyObject *self, PyObject * args)
{
    Data<OptionsGroup>* data = down_cast<Data<OptionsGroup> >( ((PyPtr<BaseData>*)self)->object );
    int index;
    if (!PyArg_ParseTuple(args, "i",&index))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    return PyString_FromString(data->getValue()[index].c_str());
}

extern "C" PyObject * OptionsGroupData_getSize(PyObject *self, PyObject *)
{
    Data<OptionsGroup>* data = down_cast<Data<OptionsGroup> >( ((PyPtr<BaseData>*)self)->object );
    return PyInt_FromLong((long)data->getValue().size());
}



SP_CLASS_ATTRS_BEGIN(OptionsGroupData)
SP_CLASS_ATTR(OptionsGroupData,currentItem)
SP_CLASS_ATTR(OptionsGroupData,currentId)
SP_CLASS_ATTRS_END



SP_CLASS_METHODS_BEGIN(OptionsGroupData)
SP_CLASS_METHOD(OptionsGroupData,getSize)
SP_CLASS_METHOD(OptionsGroupData,getItem)
SP_CLASS_METHOD(OptionsGroupData,getCurrentItem)
SP_CLASS_METHOD(OptionsGroupData,setCurrentItem)
SP_CLASS_METHOD(OptionsGroupData,getCurrentId)
SP_CLASS_METHOD(OptionsGroupData,setCurrentId)
SP_CLASS_METHODS_END



SP_CLASS_TYPE_PTR_ATTR(OptionsGroupData,Data<OptionsGroup>,Data)

