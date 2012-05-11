/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/defaulttype/DataTypeInfo.h>

using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;


// TODO:
// se servir du DataTypeInfo pour utiliser directement les bons type :-)


#include "Binding_BaseData.h"

extern "C" PyObject * BaseData_getAttr_name(PyObject *self, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    return PyString_FromString(data->getName().c_str());
}
extern "C" int BaseData_setAttr_name(PyObject *self, PyObject * args, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    char *str = PyString_AsString(args); // pour les setters, un seul objet et pas un tuple....
    data->setName(str);
    return 0;
}

extern "C" PyObject * BaseData_getAttr_value(PyObject *self, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast

    // on s'occupe du type
    printf("BaseData_getAttr_value typestring=%s\n",data->getValueString().c_str());

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    if (typeinfo->Text())
    {
        // it's some text
        printf("data type=text\n");
        //  return PyString_FromString(data->)
    }

    return PyString_FromString(data->getValueString().c_str());
}
extern "C" int BaseData_setAttr_value(PyObject *self, PyObject * args, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    // de quel type est args ?
    PyTypeObject *type = args->ob_type;
    bool isInt = PyType_IsSubtype(type,&PyInt_Type);
    bool isScalar = PyType_IsSubtype(type,&PyFloat_Type);
    bool isString = PyType_IsSubtype(type,&PyString_Type);
    printf ("isInt=%d\n", isInt);
    printf ("isScalar=%d\n", isScalar);
    printf ("isString=%d\n", isString);
    if (PyType_IsSubtype(type,&PyFloat_Type))
    {
        // it's a scalar
    }
    else if (PyType_IsSubtype(type,&PyFloat_Type))
    {
        // it's a scalar
    }
    else if (PyType_IsSubtype(type,&PyFloat_Type))
    {
        // it's a scalar
    }

    //
    char *str = PyString_AsString(args); // pour les setters, un seul objet et pas un tuple....
    //data->read(str);
    return 0;
}

extern "C" PyObject * BaseData_getValueTypeString(PyObject *self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    return PyString_FromString(data->getValueTypeString().c_str());
}

SP_CLASS_METHODS_BEGIN(BaseData)
SP_CLASS_METHOD(BaseData,getValueTypeString)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(BaseData)
SP_CLASS_ATTR(BaseData,name)
//SP_CLASS_ATTR(BaseData,owner)
SP_CLASS_ATTR(BaseData,value)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR(BaseData)
