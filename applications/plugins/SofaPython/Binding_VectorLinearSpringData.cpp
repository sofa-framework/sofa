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
#include "Binding_VectorLinearSpringData.h"
#include "Binding_Data.h"
using sofa::core::objectmodel::Data ;

#include <SofaDeformable/SpringForceField.h>
using sofa::component::interactionforcefield::LinearSpring ;

#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo ;

#include "Binding_LinearSpring.h"
#include "PythonToSofa.inl"

typedef LinearSpring<SReal> MyLinearSpring;
typedef sofa::helper::vector<MyLinearSpring> VectorLinearSpring;
typedef Data<VectorLinearSpring> DataBinding_VectorLinearSpring;


/// getting a Data<VectorLinearSpring>* from a PyObject*
static inline Data<VectorLinearSpring>* get_DataVectorLinearSpring(PyObject* obj) {
    return sofa::py::unwrap<Data<VectorLinearSpring>>(obj);
}


SP_CLASS_ATTR_GET(VectorLinearSpringData,value)(PyObject *self, void*)
{
    DataBinding_VectorLinearSpring* data  = get_DataVectorLinearSpring( self );

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();
    int rowWidth = typeinfo->size();
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

    if (typeinfo->size(valueVoidPtr)==1)
    {
        // this type is NOT a vector; return directly the proper native type
        const LinearSpring<SReal>& value = data->getValue()[0];
        LinearSpring<SReal> *obj = new LinearSpring<SReal>(value.m1,value.m2,value.ks,value.kd,value.initpos);
        return SP_BUILD_PYPTR(LinearSpring,LinearSpring<SReal>,obj,true); // "true", because I manage the deletion myself
    }
    else
    {
        PyObject *rows = PyList_New(nbRows);
        for (int i=0; i<nbRows; i++)
        {
            PyObject *row = PyList_New(rowWidth);
            for (int j=0; j<rowWidth; j++)
            {
                // build each value of the list
                const LinearSpring<SReal>& value = data->getValue()[i*rowWidth+j];
                LinearSpring<SReal> *obj = new LinearSpring<SReal>(value.m1,value.m2,value.ks,value.kd,value.initpos);
                PyList_SetItem(row,j,SP_BUILD_PYPTR(LinearSpring,LinearSpring<SReal>,obj,true));
            }
            PyList_SetItem(rows,i,row);
        }
        return rows;
    }
}


SP_CLASS_ATTR_SET(VectorLinearSpringData,value)(PyObject *self, PyObject * args, void*)
{
    DataBinding_VectorLinearSpring* data  = get_DataVectorLinearSpring( self );

    // string
    if (PyString_Check(args))
    {
        char *str = PyString_AsString(args); // for setters, only one object and not a tuple....

        if( strlen(str)>0u && str[0]=='@' ) // DataLink
        {
            data->setParent(str);
            data->setDirtyOutputs(); // forcing children updates (should it be done in BaseData?)
        }
        else
        {
            data->read(str);
        }
        return true;
    }

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); // info about the data value
    const bool valid = (typeinfo && typeinfo->ValidInfo());

    const int rowWidth = valid ? typeinfo->size() : 1;
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

    if( !PyList_Check(args) )
    {
        // one value
        // check the python object type
        if (rowWidth*nbRows<1 || !PyObject_IsInstance(args,reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(LinearSpring))))
        {
            // type mismatch or too long list
            PyErr_BadArgument();
            return -1;
        }

        // right number if rows ?
        if (1!=nbRows)
        {
            // only a warning; do not raise an exception...
            SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (incorrect rows count)" )
        }

        LinearSpring<SReal>* obj = dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)args)->object);
        VectorLinearSpring* vectorLinearSpring = data->beginEdit();

        (*vectorLinearSpring)[0].m1 = obj->m1;
        (*vectorLinearSpring)[0].m2 = obj->m2;
        (*vectorLinearSpring)[0].ks = obj->ks;
        (*vectorLinearSpring)[0].kd = obj->kd;
        (*vectorLinearSpring)[0].initpos = obj->initpos;

        data->endEdit();

        return 0;
    }
    else
    {
        // values list
        // is it a double-dimension list ?
        if (PyList_Check(PyList_GetItem(args,0)))
        {
            // two-dimension array!
            // right number if rows ?
            if (PyList_Size(args)!=nbRows)
            {
                // only a warning; do not raise an exception...
                SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (incorrect rows count)" )
                        if (PyList_Size(args)<nbRows)
                        nbRows = PyList_Size(args);
            }

            VectorLinearSpring* vectorLinearSpring = data->beginEdit();

            // let's fill our rows!
            for (int i=0; i<nbRows; i++)
            {
                PyObject *row = PyList_GetItem(args,i);

                // right number if list members ?
                int size = rowWidth;
                if (PyList_Size(row)!=size)
                {
                    // only a warning; do not raise an exception...
                    SP_MESSAGE_WARNING( "row "<<i<<" size mismatch for data \""<<data->getName()<<"\" (src="<<(int)PyList_Size(row)<<"x"<<nbRows<<" dst="<<size<<"x"<<nbRows<<")" )
                            if (PyList_Size(row)<size)
                            size = PyList_Size(row);
                }

                // okay, let's set our list...
                for (int j=0; j<size; j++)
                {
                    PyObject *listElt = PyList_GetItem(row,j);
                    if(!PyObject_IsInstance(listElt,reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(LinearSpring))))
                    {
                        // type mismatch
                        data->endEdit();
                        PyErr_BadArgument();
                        return -1;
                    }
                    LinearSpring<SReal>* spring=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)listElt)->object);

                    (*vectorLinearSpring)[j+i*rowWidth].m1 = spring->m1;
                    (*vectorLinearSpring)[j+i*rowWidth].m2 = spring->m2;
                    (*vectorLinearSpring)[j+i*rowWidth].ks = spring->ks;
                    (*vectorLinearSpring)[j+i*rowWidth].kd = spring->kd;
                    (*vectorLinearSpring)[j+i*rowWidth].initpos = spring->initpos;
                }
            }
            data->endEdit();
            return 0;
        }
        else
        {
            // it is a one-dimension only array
            // right number if list members ?
            int size = rowWidth*nbRows;
            if (PyList_Size(args)!=size)
            {
                // only a warning; do not raise an exception...
                SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (src="<<(int)PyList_Size(args)<<" dst="<<size<<")" )
                        if (PyList_Size(args)<size)
                        size = PyList_Size(args);
            }

            VectorLinearSpring* vectorLinearSpring = data->beginEdit();

            // okay, let's set our list...
            for (int i=0; i<size; i++)
            {
                PyObject *listElt = PyList_GetItem(args,i);
                if(!PyObject_IsInstance(listElt,reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(LinearSpring))))
                {
                    // type mismatch
                    data->endEdit();
                    PyErr_BadArgument();
                    return -1;
                }

                LinearSpring<SReal>* spring=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)listElt)->object);
                (*vectorLinearSpring)[i].m1 = spring->m1;
                (*vectorLinearSpring)[i].m2 = spring->m2;
                (*vectorLinearSpring)[i].ks = spring->ks;
                (*vectorLinearSpring)[i].kd = spring->kd;
                (*vectorLinearSpring)[i].initpos = spring->initpos;
            }
            data->endEdit();
            return 0;
        }
    }
}


static Py_ssize_t VectorLinearSpringData_length(PyObject *self)
{
    DataBinding_VectorLinearSpring* data  = get_DataVectorLinearSpring( self );
    return data->getValue().size();
}


static PyObject * VectorLinearSpringData_getitem(PyObject *self, PyObject *i)
{
    DataBinding_VectorLinearSpring* data  = get_DataVectorLinearSpring( self );

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();
    //    int rowWidth = typeinfo->size();
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
    long index = PyInt_AsLong(i);
    if (typeinfo->size(valueVoidPtr)==1)
    {
        SP_MESSAGE_WARNING( "the VectorLinearSpringData contains only one element" )

                // this type is NOT a vector; return directly the proper native type
                const LinearSpring<SReal>& value = data->getValue()[0];
        LinearSpring<SReal> *obj = new LinearSpring<SReal>(value.m1,value.m2,value.ks,value.kd,value.initpos);
        return SP_BUILD_PYPTR(LinearSpring,LinearSpring<SReal>,obj,true); // "true", because I manage the deletion myself
    }
    else
    {
        if( index>=nbRows )
        {
            SP_MESSAGE_ERROR( "the VectorLinearSpringData contains only "<<nbRows<<" element" )
                    PyErr_BadArgument();
            return NULL;
        }

        const LinearSpring<SReal>& value = data->getValue()[index];
        LinearSpring<SReal> *obj = new LinearSpring<SReal>(value.m1,value.m2,value.ks,value.kd,value.initpos);
        return SP_BUILD_PYPTR(LinearSpring,LinearSpring<SReal>,obj,true); // "true", because I manage the deletion myself
    }
}


static int VectorLinearSpringData_setitem(PyObject *self, PyObject* i, PyObject* v)
{
    DataBinding_VectorLinearSpring* data  = get_DataVectorLinearSpring( self );

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

    long index = PyInt_AsLong(i);
    if( index>=nbRows )
    {
        SP_MESSAGE_ERROR( "the VectorLinearSpringData contains only "<<nbRows<<" element" )
                PyErr_BadArgument();
        return -1;
    }

    MyLinearSpring* value= sofa::py::unwrap<MyLinearSpring>(v); // TODO: check dynamic cast
    VectorLinearSpring& vec = *data->beginEdit();

    vec[index] = *value;
    data->endEdit();

    return 0;
}


SP_CLASS_ATTRS_BEGIN(VectorLinearSpringData)
SP_CLASS_ATTR(VectorLinearSpringData,value)
SP_CLASS_ATTRS_END


SP_CLASS_METHODS_BEGIN(VectorLinearSpringData)
SP_CLASS_METHODS_END

SP_CLASS_MAPPING(VectorLinearSpringData)

SP_CLASS_TYPE_PTR_ATTR_MAPPING(VectorLinearSpringData, sofa::core::objectmodel::BaseData, Data)

