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
#include <sofa/core/objectmodel/Data.h>

using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;


// TODO:
// se servir du DataTypeInfo pour utiliser directement les bons type :-)
// Il y a un seul type "Data" exposé en python, le transtypage est géré automatiquement

#include "Binding_Data.h"

extern "C" PyObject * Data_getAttr_name(PyObject *self, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    return PyString_FromString(data->getName().c_str());
}
extern "C" int Data_setAttr_name(PyObject *self, PyObject * args, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    char *str = PyString_AsString(args); // pour les setters, un seul objet et pas un tuple....
    data->setName(str);
    return 0;
}

PyObject *GetDataValuePython(BaseData* data)
{
    // depending on the data type, we return the good python type (int, float, sting, array, ...)

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    if (typeinfo->size()==1)
    {
        // this type is NOT a vector; return directly the proper native type
        if (typeinfo->Text())
        {
            // it's some text
            return PyString_FromString(typeinfo->getTextValue(data->getValueVoidPtr(),0).c_str());
        }
        if (typeinfo->Scalar())
        {
            // it's a SReal
            return PyFloat_FromDouble(typeinfo->getScalarValue(data->getValueVoidPtr(),0));
        }
        if (typeinfo->Integer())
        {
            // it's some Integer...
            return PyInt_FromLong(typeinfo->getIntegerValue(data->getValueVoidPtr(),0));
        }
    }
    else
    {
        // this is a vector; return a python list of the corrsponding type (ints, scalars or strings)
        PyObject *list = PyList_New(typeinfo->size());
        for (unsigned int i=0; i<typeinfo->size(); i++)
        {
            // build each value of the list
            if (typeinfo->Text())
            {
                // it's some text
                PyList_SetItem(list,i,PyString_FromString(typeinfo->getTextValue(data->getValueVoidPtr(),i).c_str()));
            }
            if (typeinfo->Scalar())
            {
                // it's a SReal
                PyList_SetItem(list,i,PyFloat_FromDouble(typeinfo->getScalarValue(data->getValueVoidPtr(),i)));
            }
            if (typeinfo->Integer())
            {
                // it's some Integer...
                PyList_SetItem(list,i,PyInt_FromLong(typeinfo->getIntegerValue(data->getValueVoidPtr(),i)));
            }
        }

        return list;
    }
    // default (should not happen)...
    printf("<SofaPython> BaseData_getAttr_value WARNING: unsupported native type=%s ; returning string value\n",data->getValueTypeString().c_str());
    return PyString_FromString(data->getValueString().c_str());
}

bool SetDataValuePython(BaseData* data, PyObject* args)
{
    // de quel type est args ?
    bool isInt = PyInt_Check(args);
    bool isScalar = PyFloat_Check(args);
    bool isString = PyString_Check(args);
    bool isList = PyList_Check(args);
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); // info about the data value
    if (isInt)
    {
        // it's an int

        if (typeinfo->size()<1 || (!typeinfo->Integer() && !typeinfo->Scalar()))
        {
            // type mismatch or too long list
            PyErr_BadArgument();
            return false;
        }
        long value = PyInt_AsLong(args);
        if (typeinfo->Scalar())
            typeinfo->setScalarValue((void*)data->getValueVoidPtr(),0,(SReal)value); // cast int to float
        else
            typeinfo->setIntegerValue((void*)data->getValueVoidPtr(),0,value);
        return true;
    }
    else if (isScalar)
    {
        // it's a scalar
        if (typeinfo->size()<1 || !typeinfo->Scalar())
        {
            // type mismatch or too long list
            PyErr_BadArgument();
            return false;
        }
        SReal value = PyFloat_AsDouble(args);
        typeinfo->setScalarValue((void*)data->getValueVoidPtr(),0,value);
        return true;
    }
    else if (isString)
    {
        // it's a string
        char *str = PyString_AsString(args); // pour les setters, un seul objet et pas un tuple....
        data->read(str);
        return true;
    }
    else if (isList)
    {
        // it's a list
        // check list emptyness
        if (PyList_Size(args)==0)
        {
            // empty list: ignored
            return true;
        }

        // right number if list members ?
        int size = typeinfo->size();
        if (PyList_Size(args)!=typeinfo->size())
        {
            // only a warning; do not raise an exception...
            printf("<SofaPython> Warning: list size mismatch for data \"%s\"\n",data->getName().c_str());
            if (PyList_Size(args)<typeinfo->size())
                size = PyList_Size(args);
        }

        // okay, let's set our list...
        for (int i=0; i<size; i++)
        {
            PyObject *listElt = PyList_GetItem(args,i);

            if (PyInt_Check(listElt))
            {
                // it's an int
                if (typeinfo->Integer())
                {
                    // integer value
                    long value = PyInt_AsLong(listElt);
                    typeinfo->setIntegerValue((void*)data->getValueVoidPtr(),i,value);
                }
                else if (typeinfo->Scalar())
                {
                    // cast to scalar value
                    SReal value = (SReal)PyInt_AsLong(listElt);
                    typeinfo->setScalarValue((void*)data->getValueVoidPtr(),i,value);
                }
                else
                {
                    // type mismatch
                    PyErr_BadArgument();
                    return false;
                }
            }
            else if (PyFloat_Check(listElt))
            {
                // it's a scalar
                if (!typeinfo->Scalar())
                {
                    // type mismatch
                    PyErr_BadArgument();
                    return false;
                }
                SReal value = PyFloat_AsDouble(listElt);
                typeinfo->setScalarValue((void*)data->getValueVoidPtr(),i,value);
            }
            else if (PyString_Check(listElt))
            {
                // it's a string
                if (!typeinfo->Text())
                {
                    // type mismatch
                    PyErr_BadArgument();
                    return false;
                }
                char *str = PyString_AsString(listElt); // pour les setters, un seul objet et pas un tuple....
                typeinfo->setTextValue((void*)data->getValueVoidPtr(),i,str);
            }
            else
            {
                printf("Lists not yet supported...\n");
                PyErr_BadArgument();
                return false;

            }
        }

        return true;
    }

    return false;

}



extern "C" PyObject * Data_getAttr_value(PyObject *self, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    return GetDataValuePython(data);
}

extern "C" int Data_setAttr_value(PyObject *self, PyObject * args, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    if (SetDataValuePython(data,args))
        return 0;   // OK


    printf("<SofaPython> argument type not supported\n");
    PyErr_BadArgument();
    return -1;
}

// access ONE element of the vector
extern "C" PyObject * Data_getValue(PyObject *self, PyObject * args)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); // info about the data value
    int index;
    if (!PyArg_ParseTuple(args, "i",&index))
    {
        PyErr_BadArgument();
        return 0;
    }
    if ((unsigned int)index>=typeinfo->size())
    {
        // out of bounds!
        printf("<SofaPython> Error: Data.getValue index overflow\n");
        PyErr_BadArgument();
        return 0;
    }
    if (typeinfo->Scalar())
        return PyFloat_FromDouble(typeinfo->getScalarValue(data->getValueVoidPtr(),index));
    if (typeinfo->Integer())
        return PyInt_FromLong(typeinfo->getIntegerValue(data->getValueVoidPtr(),index));
    if (typeinfo->Text())
        return PyString_FromString(typeinfo->getTextValue(data->getValueVoidPtr(),index).c_str());

    // should never happen....
    printf("<SofaPython> Error: Data.getValue unknown data type\n");
    PyErr_BadArgument();
    return 0;
}
extern "C" PyObject * Data_setValue(PyObject *self, PyObject * args)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); // info about the data value
    int index;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "iO",&index,&value))
    {
        PyErr_BadArgument();
        return 0;
    }
    if ((unsigned int)index>=typeinfo->size())
    {
        // out of bounds!
        printf("<SofaPython> Error: Data.setValue index overflow\n");
        PyErr_BadArgument();
        return 0;
    }
    if (typeinfo->Scalar() && PyFloat_Check(value))
    {
        typeinfo->setScalarValue((void*)data->getValueVoidPtr(),index,PyFloat_AsDouble(value));
        return PyInt_FromLong(0);
    }
    if (typeinfo->Integer() && PyInt_Check(value))
    {
        typeinfo->setIntegerValue((void*)data->getValueVoidPtr(),index,PyInt_AsLong(value));
        return PyInt_FromLong(0);
    }
    if (typeinfo->Text() && PyString_Check(value))
    {
        typeinfo->setTextValue((void*)data->getValueVoidPtr(),index,PyString_AsString(value));
        return PyInt_FromLong(0);
    }

    // should never happen....
    printf("<SofaPython> Error: Data.setValue type mismatch\n");
    PyErr_BadArgument();
    return 0;
}


extern "C" PyObject * Data_getValueTypeString(PyObject *self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;
    return PyString_FromString(data->getValueTypeString().c_str());
}

extern "C" PyObject * Data_getValueString(PyObject *self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;
    return PyString_FromString(data->getValueString().c_str());
}

SP_CLASS_METHODS_BEGIN(Data)
SP_CLASS_METHOD(Data,getValueTypeString)
SP_CLASS_METHOD(Data,getValueString)
SP_CLASS_METHOD(Data,setValue)
SP_CLASS_METHOD(Data,getValue)
SP_CLASS_METHODS_END


SP_CLASS_ATTRS_BEGIN(Data)
SP_CLASS_ATTR(Data,name)
//SP_CLASS_ATTR(BaseData,owner)
SP_CLASS_ATTR(Data,value)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR(Data,BaseData)
