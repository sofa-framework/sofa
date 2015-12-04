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
#include "Binding_Data.h"
#include "Binding_LinearSpring.h"

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/BaseNode.h>


#include <sofa/core/visual/DisplayFlags.h>
#include "Binding_DisplayFlagsData.h"

#include <sofa/helper/OptionsGroup.h>
#include "Binding_OptionsGroupData.h"

#include <SofaDeformable/SpringForceField.h>

using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;
using namespace sofa::component::interactionforcefield;


// TODO:
// se servir du DataTypeInfo pour utiliser directement les bons type :-)
// Il y a un seul type "Data" exposé en python, le transtypage est géré automatiquement


SP_CLASS_ATTR_GET(Data,name)(PyObject *self, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    return PyString_FromString(data->getName().c_str());
}
SP_CLASS_ATTR_SET(Data,name)(PyObject *self, PyObject * args, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    char *str = PyString_AsString(args); // for setters, only one object and not a tuple....
    data->setName(str);
    return 0;
}

PyObject *GetDataValuePython(BaseData* data)
{
    // depending on the data type, we return the good python type (int, float, string, array, ...)

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();
    int rowWidth = typeinfo->size();
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

    // special cases...
    if( Data<sofa::core::visual::DisplayFlags>* df = dynamic_cast<Data<sofa::core::visual::DisplayFlags>*>(data) )
    {
        return SP_BUILD_PYPTR(DisplayFlagsData,BaseData,df,false);
    }
    else if( Data<sofa::helper::OptionsGroup>* og = dynamic_cast<Data<sofa::helper::OptionsGroup>*>(data) )
    {
        return SP_BUILD_PYPTR(OptionsGroupData,BaseData,og,false);
    }
    else if ( Data<sofa::helper::vector<LinearSpring<SReal> > >* vectorLinearSpring = dynamic_cast<Data<sofa::helper::vector<LinearSpring<SReal> > >*>(data) )
    {
        // special type, a vector of LinearSpring objects
        if (typeinfo->size(valueVoidPtr)==1)
        {
            // this type is NOT a vector; return directly the proper native type
            const LinearSpring<SReal> value = vectorLinearSpring->getValue()[0];
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
                    const LinearSpring<SReal> value = vectorLinearSpring->getValue()[i*rowWidth+j];
                    LinearSpring<SReal> *obj = new LinearSpring<SReal>(value.m1,value.m2,value.ks,value.kd,value.initpos);
                    PyList_SetItem(row,j,SP_BUILD_PYPTR(LinearSpring,LinearSpring<SReal>,obj,true));
                }
                PyList_SetItem(rows,i,row);
            }

            return rows;
        }

    }

    if (!typeinfo->Container())
    {
        // this type is NOT a vector; return directly the proper native type
        if (typeinfo->Text())
        {
            // it's some text
            return PyString_FromString(typeinfo->getTextValue(valueVoidPtr,0).c_str());
        }
        if (typeinfo->Scalar())
        {
            // it's a SReal
            return PyFloat_FromDouble(typeinfo->getScalarValue(valueVoidPtr,0));
        }
        if (typeinfo->Integer())
        {
            // it's some Integer...
            return PyInt_FromLong((long)typeinfo->getIntegerValue(valueVoidPtr,0));
        }

        // this type is not yet supported
        SP_MESSAGE_WARNING( "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" )
        return PyString_FromString(data->getValueString().c_str());
    }
    else
    {
        // this is a vector; return a python list of the corresponding type (ints, scalars or strings)

        if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
        {
            SP_MESSAGE_WARNING( "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" )
            return PyString_FromString(data->getValueString().c_str());
        }

        PyObject *rows = PyList_New(nbRows);
        for (int i=0; i<nbRows; i++)
        {
            PyObject *row = PyList_New(rowWidth);
            for (int j=0; j<rowWidth; j++)
            {
                // build each value of the list
                if (typeinfo->Text())
                {
                    // it's some text
                    PyList_SetItem(row,j,PyString_FromString(typeinfo->getTextValue(valueVoidPtr,i*rowWidth+j).c_str()));
                }
                else if (typeinfo->Scalar())
                {
                    // it's a Real
                    PyList_SetItem(row,j,PyFloat_FromDouble(typeinfo->getScalarValue(valueVoidPtr,i*rowWidth+j)));
                }
                else if (typeinfo->Integer())
                {
                    // it's some Integer...
                    PyList_SetItem(row,j,PyInt_FromLong((long)typeinfo->getIntegerValue(valueVoidPtr,i*rowWidth+j)));
                }
                else
                {
                    // this type is not yet supported (should not happen)
                    SP_MESSAGE_ERROR( "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value (should not come here!)" )
                }
            }
            PyList_SetItem(rows,i,row);
        }

        return rows;
    }

    // default (should not happen)...
    SP_MESSAGE_WARNING( "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value (should not come here!)" )
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
    int rowWidth = (typeinfo && typeinfo->ValidInfo()) ? typeinfo->size() : 1;
    int nbRows = (typeinfo && typeinfo->ValidInfo()) ? typeinfo->size(data->getValueVoidPtr()) / typeinfo->size() : 1;

    // special cases...
    Data<sofa::helper::vector<LinearSpring<SReal> > >* dataVectorLinearSpring = dynamic_cast<Data<sofa::helper::vector<LinearSpring<SReal> > >*>(data);
    if (dataVectorLinearSpring)
    {
        // special type, a vector of LinearSpring objects

        if (!isList)
        {
            // one value
            // check the python object type
            if (rowWidth*nbRows<1 || !PyObject_IsInstance(args,reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(LinearSpring))))
            {
                // type mismatch or too long list
                PyErr_BadArgument();
                return false;
            }

            LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)args)->object);
            sofa::helper::vector<LinearSpring<SReal> >* vectorLinearSpring = dataVectorLinearSpring->beginEdit();

            (*vectorLinearSpring)[0].m1 = obj->m1;
            (*vectorLinearSpring)[0].m2 = obj->m2;
            (*vectorLinearSpring)[0].ks = obj->ks;
            (*vectorLinearSpring)[0].kd = obj->kd;
            (*vectorLinearSpring)[0].initpos = obj->initpos;

            dataVectorLinearSpring->endEdit();

            return true;
        }
        else
        {
            // values list
            // is it a double-dimension list ?
            //PyObject *firstRow = PyList_GetItem(args,0);

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

                sofa::helper::vector<LinearSpring<SReal> >* vectorLinearSpring = dataVectorLinearSpring->beginEdit();

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
                            dataVectorLinearSpring->endEdit();
                            PyErr_BadArgument();
                            return false;
                        }
                        LinearSpring<SReal>* spring=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)listElt)->object);


                        (*vectorLinearSpring)[j+i*rowWidth].m1 = spring->m1;
                        (*vectorLinearSpring)[j+i*rowWidth].m2 = spring->m2;
                        (*vectorLinearSpring)[j+i*rowWidth].ks = spring->ks;
                        (*vectorLinearSpring)[j+i*rowWidth].kd = spring->kd;
                        (*vectorLinearSpring)[j+i*rowWidth].initpos = spring->initpos;

                    }



                }

                dataVectorLinearSpring->endEdit();

                return true;

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

                sofa::helper::vector<LinearSpring<SReal> >* vectorLinearSpring = dataVectorLinearSpring->beginEdit();

                // okay, let's set our list...
                for (int i=0; i<size; i++)
                {

                    PyObject *listElt = PyList_GetItem(args,i);

                    if(!PyObject_IsInstance(listElt,reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(LinearSpring))))
                    {
                        // type mismatch
                        dataVectorLinearSpring->endEdit();
                        PyErr_BadArgument();
                        return false;
                    }

                    LinearSpring<SReal>* spring=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)listElt)->object);

                    (*vectorLinearSpring)[i].m1 = spring->m1;
                    (*vectorLinearSpring)[i].m2 = spring->m2;
                    (*vectorLinearSpring)[i].ks = spring->ks;
                    (*vectorLinearSpring)[i].kd = spring->kd;
                    (*vectorLinearSpring)[i].initpos = spring->initpos;


    /*
                    if (PyFloat_Check(listElt))
                    {
                        // it's a scalar
                        if (!typeinfo->Scalar())
                        {
                            // type mismatch
                            PyErr_BadArgument();
                            return false;
                        }
                        SReal value = PyFloat_AsDouble(listElt);
                        void* editVoidPtr = data->beginEditVoidPtr();
                        typeinfo->setScalarValue(editVoidPtr,i,value);
                        data->endEditVoidPtr();
                    }
     */
                }
                dataVectorLinearSpring->endEdit();

                return true;
            }
        }


        return false;
    }


    if (isInt)
    {
        // it's an int

        if (rowWidth*nbRows<1 || (!typeinfo->Integer() && !typeinfo->Scalar()))
        {
            // type mismatch or too long list
            PyErr_BadArgument();
            return false;
        }
        long value = PyInt_AsLong(args);
        void* editVoidPtr = data->beginEditVoidPtr();
        if (typeinfo->Scalar())
            typeinfo->setScalarValue(editVoidPtr,0,(SReal)value); // cast int to float
        else
            typeinfo->setIntegerValue(editVoidPtr,0,value);
        data->endEditVoidPtr();
        return true;
    }
    else if (isScalar)
    {
        // it's a scalar
        if (rowWidth*nbRows<1 || !typeinfo->Scalar())
        {
            // type mismatch or too long list
            PyErr_BadArgument();
            return false;
        }
        SReal value = PyFloat_AsDouble(args);
        void* editVoidPtr = data->beginEditVoidPtr();
        typeinfo->setScalarValue(editVoidPtr,0,value);
        data->endEditVoidPtr();
        return true;
    }
    else if (isString)
    {
        // it's a string
        char *str = PyString_AsString(args); // for setters, only one object and not a tuple....

        if( strlen(str)>0 && str[0]=='@' ) // DataLink
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
    else if (isList)
    {
        // it's a list
        // check list emptyness
        if (PyList_Size(args)==0)
        {
            data->read("");
            return true;
        }

        // is it a double-dimension list ?
        //PyObject *firstRow = PyList_GetItem(args,0);

        if (PyList_Check(PyList_GetItem(args,0)))
        {
            // two-dimension array!

            void* editVoidPtr = data->beginEditVoidPtr();

            // same number of rows?
            {
            int newNbRows = PyList_Size(args);
            if (newNbRows!=nbRows)
            {
                // try to resize (of course, it is not possible with every container, the resize policy is defined in DataTypeInfo)
                typeinfo->setSize( editVoidPtr, newNbRows*rowWidth );

                if( typeinfo->size(editVoidPtr) != (size_t)(newNbRows*rowWidth) )
                {
                    // resizing was not possible
                    // only a warning; do not raise an exception...
                    SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (incorrect rows count)" )
                    if (newNbRows<nbRows)
                        nbRows = newNbRows;
                }
                else
                {
                    // resized
                    nbRows = newNbRows;
                }
            }
            }


            // let's fill our rows!
            for (int i=0; i<nbRows; i++)
            {
                PyObject *row = PyList_GetItem(args,i);

                // right number of list members ?
                int size = rowWidth;
                if (PyList_Size(row)!=size)
                {
                    // only a warning; do not raise an exception...
                    SP_MESSAGE_WARNING( "row "<<i<<" size mismatch for data \""<<data->getName()<<"\"" )
                    if (PyList_Size(row)<size)
                        size = PyList_Size(row);
                }

                // okay, let's set our list...
                for (int j=0; j<size; j++)
                {

                    PyObject *listElt = PyList_GetItem(row,j);

                    if (PyInt_Check(listElt))
                    {
                        // it's an int
                        if (typeinfo->Integer())
                        {
                            // integer value
                            long value = PyInt_AsLong(listElt);
                            typeinfo->setIntegerValue(editVoidPtr,i*rowWidth+j,value);
                        }
                        else if (typeinfo->Scalar())
                        {
                            // cast to scalar value
                            SReal value = (SReal)PyInt_AsLong(listElt);
                            typeinfo->setScalarValue(editVoidPtr,i*rowWidth+j,value);
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
                        typeinfo->setScalarValue(editVoidPtr,i*rowWidth+j,value);
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
                        typeinfo->setTextValue(editVoidPtr,i*rowWidth+j,str);
                    }
                    else
                    {
                        printf("Lists not yet supported...\n");
                        PyErr_BadArgument();
                        return false;

                    }
                }



            }
            data->endEditVoidPtr();
            return true;

        }
        else
        {
            // it is a one-dimension only array

            void* editVoidPtr = data->beginEditVoidPtr();

            // same number of list members?
            int size = rowWidth*nbRows; // start with oldsize
            {
            int newSize = PyList_Size(args);
            if (newSize!=size)
            {
                // try to resize (of course, it is not possible with every container, the resize policy is defined in DataTypeInfo)
                typeinfo->setSize( editVoidPtr, newSize );

                if( typeinfo->size(editVoidPtr) != (size_t)newSize )
                {
                    // resizing was not possible
                    // only a warning; do not raise an exception...
                    SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (incorrect rows count)" )
                    if (newSize<size)
                        size = newSize;
                }
                else
                {
                    // resized
                    size = newSize;
                }
            }
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
                        typeinfo->setIntegerValue(editVoidPtr,i,value);
                    }
                    else if (typeinfo->Scalar())
                    {
                        // cast to scalar value
                        SReal value = (SReal)PyInt_AsLong(listElt);
                        typeinfo->setScalarValue(editVoidPtr,i,value);
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
                    typeinfo->setScalarValue(editVoidPtr,i,value);
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
                    typeinfo->setTextValue(editVoidPtr,i,str);
                }
                else
                {
                    printf("Lists not yet supported...\n");
                    PyErr_BadArgument();
                    return false;

                }
            }
            data->endEditVoidPtr();
            return true;
        }

    }

    return false;

}



SP_CLASS_ATTR_GET(Data,value)(PyObject *self, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    return GetDataValuePython(data);
}

SP_CLASS_ATTR_SET(Data,value)(PyObject *self, PyObject * args, void*)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object; // TODO: check dynamic cast
    if (SetDataValuePython(data,args))
        return 0;   // OK


    SP_MESSAGE_ERROR( "argument type not supported" )
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
        Py_RETURN_NONE;
    }
    if ((unsigned int)index>=typeinfo->size())
    {
        // out of bounds!
        SP_MESSAGE_ERROR( "Data.getValue index overflow" )
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    if (typeinfo->Scalar())
        return PyFloat_FromDouble(typeinfo->getScalarValue(data->getValueVoidPtr(),index));
    if (typeinfo->Integer())
        return PyInt_FromLong((long)typeinfo->getIntegerValue(data->getValueVoidPtr(),index));
    if (typeinfo->Text())
        return PyString_FromString(typeinfo->getTextValue(data->getValueVoidPtr(),index).c_str());

    // should never happen....
    SP_MESSAGE_ERROR( "Data.getValue unknown data type" )
    PyErr_BadArgument();
    Py_RETURN_NONE;
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
        Py_RETURN_NONE;
    }
    if ((unsigned int)index>=typeinfo->size())
    {
        // out of bounds!
        SP_MESSAGE_ERROR( "Data.setValue index overflow" )
        PyErr_BadArgument();
        Py_RETURN_NONE;
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
    SP_MESSAGE_ERROR( "Data.setValue type mismatch" )
    PyErr_BadArgument();
    Py_RETURN_NONE;
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

extern "C" PyObject * Data_getSize(PyObject *self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    int rowWidth = typeinfo->size();
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

    printf("Data_getSize rowWidth=%d nbRows=%d\n",rowWidth,nbRows);

    return PyInt_FromLong(0); //temp
}

extern "C" PyObject * Data_setSize(PyObject *self, PyObject * args)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;
    int size;
    if (!PyArg_ParseTuple(args, "i",&size))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    typeinfo->setSize((void*)data->getValueVoidPtr(),size);
    Py_RETURN_NONE;
}


extern "C" PyObject * Data_unset(PyObject *self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;

    data->unset();

    Py_RETURN_NONE;
}

extern "C" PyObject * Data_updateIfDirty(PyObject *self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;

    data->updateIfDirty();

    Py_RETURN_NONE;
}


extern "C" PyObject * Data_read(PyObject *self, PyObject * args)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;

    PyObject *value;
    if (!PyArg_ParseTuple(args, "O",&value))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    if (PyString_Check(value))
    {
        data->read(PyString_AsString(value));
    }
    else
    {
        SP_MESSAGE_ERROR( "Data.read type mismatch" )
        PyErr_BadArgument();
    }

    Py_RETURN_NONE;
}

extern "C" PyObject * Data_setParent(PyObject *self, PyObject * args)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;

    PyObject *value;
    if (!PyArg_ParseTuple(args, "O",&value))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    if (PyString_Check(value))
    {
        data->setParent(PyString_AsString(value));
        data->setDirtyOutputs(); // forcing children updates (should it be done in BaseData?)
    }
    else
    {
        SP_MESSAGE_ERROR( "Data.setParent type mismatch" )
        PyErr_BadArgument();
    }

    Py_RETURN_NONE;
}


// returns the complete link path name (i.e. following the shape "@/path/to/my/object.dataname")
extern "C" PyObject * Data_getLinkPath(PyObject * self, PyObject * /*args*/)
{
    BaseData* data=((PyPtr<BaseData>*)self)->object;
    Base* owner = data->getOwner();

    if( owner )
    {
        if( BaseObject* obj = owner->toBaseObject() )
            return PyString_FromString(("@"+obj->getPathName()+"."+data->getName()).c_str());
        else if( BaseNode* node = owner->toBaseNode() )
            return PyString_FromString(("@"+node->getPathName()+"."+data->getName()).c_str());
    }

    // default: no owner or owner of unknown type
    SP_MESSAGE_WARNING( "Data_getLinkName the Data has no known owner" )
    return PyString_FromString(data->getName().c_str());
}

SP_CLASS_METHODS_BEGIN(Data)
SP_CLASS_METHOD(Data,getValueTypeString)
SP_CLASS_METHOD(Data,getValueString)
SP_CLASS_METHOD(Data,setValue)
SP_CLASS_METHOD(Data,getValue)
SP_CLASS_METHOD(Data,getSize)
SP_CLASS_METHOD(Data,setSize)
SP_CLASS_METHOD(Data,unset)
SP_CLASS_METHOD(Data,updateIfDirty)
SP_CLASS_METHOD(Data,read)
SP_CLASS_METHOD(Data,setParent)
SP_CLASS_METHOD(Data,getLinkPath)
SP_CLASS_METHODS_END


SP_CLASS_ATTRS_BEGIN(Data)
SP_CLASS_ATTR(Data,name)
//SP_CLASS_ATTR(BaseData,owner)
SP_CLASS_ATTR(Data,value)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR(Data,BaseData)
