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
#include <sstream>

#include "Binding_Data.h"
#include "Binding_LinearSpring.h"

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::Base ;

#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo ;

#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include "PythonToSofa.inl"

static BaseData* get_basedata(PyObject* self) {
    return sofa::py::unwrap<BaseData>(self);
}


SP_CLASS_ATTR_GET(Data,name)(PyObject *self, void*)
{
    BaseData* data = get_basedata( self );
    return PyString_FromString(data->getName().c_str());
}
SP_CLASS_ATTR_SET(Data,name)(PyObject *self, PyObject * args, void*)
{
    BaseData* data = get_basedata( self );
    char *str = PyString_AsString(args); // for setters, only one object and not a tuple....
    data->setName(str);
    return 0;
}

PyObject *GetDataValuePython(BaseData* data)
{
    /// depending on the data type, we return the good python type (int, float, string, array, ...)
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();

    if (!typeinfo->Container())
    {
        /// this type is NOT a vector; return directly the proper native type
        if (typeinfo->Text())
        {
            return PyString_FromString(typeinfo->getTextValue(valueVoidPtr,0).c_str());
        }
        if (typeinfo->Scalar())
        {
            return PyFloat_FromDouble(typeinfo->getScalarValue(valueVoidPtr,0));
        }
        if (typeinfo->Integer())
        {
            return PyInt_FromLong((long)typeinfo->getIntegerValue(valueVoidPtr,0));
        }

        /// This type is not yet supported, the fallback scenario is to convert it using the python str() function and emit
        /// a warning message.
        msg_warning(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
        return PyString_FromString(data->getValueString().c_str());
    }
    else
    {
        int rowWidth = typeinfo->size();
        int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

        /// this is a vector; return a python list of the corresponding type (ints, scalars or strings)
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
                /// build each value of the list
                if (typeinfo->Text())
                {
                    PyList_SetItem(row,j,PyString_FromString(typeinfo->getTextValue(valueVoidPtr,i*rowWidth+j).c_str()));
                }
                else if (typeinfo->Scalar())
                {
                    PyList_SetItem(row,j,PyFloat_FromDouble(typeinfo->getScalarValue(valueVoidPtr,i*rowWidth+j)));
                }
                else if (typeinfo->Integer())
                {
                    PyList_SetItem(row,j,PyInt_FromLong((long)typeinfo->getIntegerValue(valueVoidPtr,i*rowWidth+j)));
                }
                else
                {
                    //TODO(PR:304) If this should not happen (see comment later) then we should rise an exception instead of providing a fallback scenario.
                    /// this type is not yet supported (should not happen)
                    SP_MESSAGE_ERROR( "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value (should not come here!)" )
                }
            }
            PyList_SetItem(rows,i,row);
        }

        return rows;
    }

    //TODO(PR:304) If this should not happen (see comment later) then we should rise an exception instead of providing a fallback scenario.
    /// default (should not happen)...
    SP_MESSAGE_WARNING( "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value (should not come here!)" )
    return PyString_FromString(data->getValueString().c_str());
}


static int SetDataValuePythonList(BaseData* data, PyObject* args,
                            const int rowWidth, int nbRows) {
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); // info about the data value

    /// If list is empty we can safely exit.
    if (PyList_Size(args)==0)
    {
        data->read("");
        return 0;
    }

    /// Check if the list have two dimmensions
    if (PyList_Check(PyList_GetItem(args,0)))
    {
        /// Handle the two-dimension array case.
        void* editVoidPtr = data->beginEditVoidPtr();

        /// same number of rows?
        {
            int newNbRows = PyList_Size(args);
            if (newNbRows!=nbRows)
            {
                /// try to resize (of course, it is not possible with every container, the resize policy is defined in DataTypeInfo)
                typeinfo->setSize( editVoidPtr, newNbRows*rowWidth );

                if( typeinfo->size(editVoidPtr) != (size_t)(newNbRows*rowWidth) )
                {
                    /// resizing was not possible
                    /// only a warning and not an exception because we have a fallback solution.
                    SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (incorrect rows count)" )
                        if (newNbRows<nbRows)
                            nbRows = newNbRows;
                }
                else
                {
                    /// resized
                    nbRows = newNbRows;
                }
            }
        }

        /// let's fill our rows!
        for (int i=0; i<nbRows; i++)
        {
            PyObject *row = PyList_GetItem(args,i);

            /// right number of list members ?
            int size = rowWidth;
            if (PyList_Size(row)!=size)
            {
                /// only a warning and not an exception because we have a fallback solution.
                SP_MESSAGE_WARNING( "row "<<i<<" size mismatch for data \""<<data->getName()<<"\"" )
                    if (PyList_Size(row)<size)
                        size = PyList_Size(row);
            }

            /// Analyze the list and convert it
            for (int j=0; j<size; j++)
            {

                PyObject *listElt = PyList_GetItem(row,j);

                if (PyInt_Check(listElt))
                {
                    /// it's an int
                    if (typeinfo->Integer())
                    {
                        /// integer value
                        long value = PyInt_AsLong(listElt);
                        typeinfo->setIntegerValue(editVoidPtr,i*rowWidth+j,value);
                    }
                    else if (typeinfo->Scalar())
                    {
                        /// cast to scalar value
                        SReal value = (SReal)PyInt_AsLong(listElt);
                        typeinfo->setScalarValue(editVoidPtr,i*rowWidth+j,value);
                    }
                    else
                    {
                        PyErr_BadArgument();
                        return -1;
                    }
                }
                else if (PyFloat_Check(listElt))
                {
                    /// it's a scalar
                    if (!typeinfo->Scalar())
                    {
                        PyErr_BadArgument();
                        return -1;
                    }
                    SReal value = PyFloat_AsDouble(listElt);
                    typeinfo->setScalarValue(editVoidPtr,i*rowWidth+j,value);
                }
                else if (PyString_Check(listElt))
                {
                    /// it's a string
                    if (!typeinfo->Text())
                    {
                        PyErr_BadArgument();
                        return -1;
                    }
                    char *str = PyString_AsString(listElt);
                    typeinfo->setTextValue(editVoidPtr,i*rowWidth+j,str);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "Lists not yet supported.");
                    return -1;
                }
            }
        }
        data->endEditVoidPtr();
        return 0;
    }
    else
    {
        /// it is a one-dimension only array
        void* editVoidPtr = data->beginEditVoidPtr();

        /// same number of list members?
        int size = rowWidth*nbRows; /// start with oldsize
        {
            int newSize = PyList_Size(args);
            if (newSize!=size)
            {
                /// try to resize (of course, it is not possible with every container, the resize policy is defined in DataTypeInfo)
                typeinfo->setSize( editVoidPtr, newSize );

                if( typeinfo->size(editVoidPtr) != (size_t)newSize )
                {
                    /// resizing was not possible
                    /// only a warning; do not raise an exception...
                    SP_MESSAGE_WARNING( "list size mismatch for data \""<<data->getName()<<"\" (incorrect rows count)" )
                        if (newSize<size)
                            size = newSize;
                }
                else
                {
                    /// resized
                    size = newSize;
                }
            }
        }

        /// Analyze the list and convert it
        for (int i=0; i<size; i++)
        {
            PyObject *listElt = PyList_GetItem(args,i);

            if (PyInt_Check(listElt))
            {
                if (typeinfo->Integer())
                {
                    long value = PyInt_AsLong(listElt);
                    typeinfo->setIntegerValue(editVoidPtr,i,value);
                }
                else if (typeinfo->Scalar())
                {
                    SReal value = (SReal)PyInt_AsLong(listElt);
                    typeinfo->setScalarValue(editVoidPtr,i,value);
                }
                else
                {
                    PyErr_BadArgument();
                    return -1;
                }
            }
            else if (PyFloat_Check(listElt))
            {
                if (!typeinfo->Scalar())
                {
                    PyErr_BadArgument();
                    return -1;
                }
                SReal value = PyFloat_AsDouble(listElt);
                typeinfo->setScalarValue(editVoidPtr,i,value);
            }
            else if (PyString_Check(listElt))
            {
                if (!typeinfo->Text())
                {
                    PyErr_BadArgument();
                    return -1;
                }
                char *str = PyString_AsString(listElt); /// pour les setters, un seul objet et pas un tuple....
                typeinfo->setTextValue(editVoidPtr,i,str);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "Lists not yet supported as parameter");
                return -1;
            }
        }
        data->endEditVoidPtr();
        return 0;
    }

    /// no idea whether this is reachable
    PyErr_BadArgument();
    return -1;
}



int SetDataValuePython(BaseData* data, PyObject* args)
{
    if (PyString_Check(args))
    {
        char *str = PyString_AsString(args); /// for setters, only one object and not a tuple....

        if( strlen(str)>0u && str[0]=='@' )  /// DataLink
        {
            data->setParent(str);
            data->setDirtyOutputs();         /// forcing children updates (should it be done in BaseData?)
        }
        else
        {
            data->read(str);
        }
        return 0;
    }

    // Unicode
    if (PyUnicode_Check(args))
    {
        std::stringstream streamstr;
        PyObject* tmpstr = PyUnicode_AsUTF8String(args);
        streamstr << PyString_AsString(tmpstr) ;
        Py_DECREF(tmpstr);
        std::string str(streamstr.str());

        if( str.size() > 0u && str[0]=='@' ) // DataLink
        {
            data->setParent(str);
            data->setDirtyOutputs(); // forcing children updates (should it be done in BaseData?)
        }
        else
        {
            data->read(str);
        }
        return 0;
    }
    /// Get the info about the data value through the introspection mechanism.
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const bool valid = (typeinfo && typeinfo->ValidInfo());

    const int rowWidth = valid ? typeinfo->size() : 1;
    const int nbRows = valid ? typeinfo->size(data->getValueVoidPtr()) / typeinfo->size() : 1;

    if (PyInt_Check(args))
    {
        if (rowWidth*nbRows<1 || (!typeinfo->Integer() && !typeinfo->Scalar()))
        {
            /// type mismatch or too long list
            PyErr_BadArgument();
            return -1;
        }
        long value = PyInt_AsLong(args);
        void* editVoidPtr = data->beginEditVoidPtr();
        if (typeinfo->Scalar())
            typeinfo->setScalarValue(editVoidPtr,0,(SReal)value); /// cast int to float
        else
            typeinfo->setIntegerValue(editVoidPtr,0,value);
        data->endEditVoidPtr();
        return 0;
    }

    if (PyFloat_Check(args))
    {
        if (rowWidth*nbRows<1 || !typeinfo->Scalar())
        {
            /// type mismatch or too long list
            PyErr_BadArgument();
            return -1;
        }
        SReal value = PyFloat_AsDouble(args);
        void* editVoidPtr = data->beginEditVoidPtr();
        typeinfo->setScalarValue(editVoidPtr,0,value);
        data->endEditVoidPtr();
        return 0;
    }

    if ( PyList_Check(args))
    {
        return SetDataValuePythonList(data, args, rowWidth, nbRows);
    }

    /// BaseData
    if( BaseData* targetData = get_basedata(args) )
    {
        // TODO improve data to data copy
        SP_MESSAGE_WARNING( "Data to Data copy is using string serialization for now. This may results in poor performances." );
        data->read( targetData->getValueString() );
        return 0;
    }

    PyErr_BadArgument();
    return -1;
}


SP_CLASS_ATTR_GET(Data,value)(PyObject *self, void*)
{
    BaseData* data = get_basedata( self );
    return GetDataValuePython(data);
}


SP_CLASS_ATTR_SET(Data,value)(PyObject *self, PyObject * args, void*)
{
    BaseData* data = get_basedata( self );
    return SetDataValuePython(data,args);
}


/// access ONE element of the vector
static PyObject * Data_getValue(PyObject *self, PyObject * args)
{
    BaseData* data = get_basedata( self );
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); /// info about the data value
    int index;
    if (!PyArg_ParseTuple(args, "i",&index))
    {
        return NULL;
    }
    if ((unsigned int)index>=typeinfo->size())
    {
        /// out of bounds!
        SP_MESSAGE_ERROR( "Data.getValue index overflow" )

                PyErr_BadArgument();
        return NULL;
    }
    if (typeinfo->Scalar())
        return PyFloat_FromDouble(typeinfo->getScalarValue(data->getValueVoidPtr(),index));
    if (typeinfo->Integer())
        return PyInt_FromLong((long)typeinfo->getIntegerValue(data->getValueVoidPtr(),index));
    if (typeinfo->Text())
        return PyString_FromString(typeinfo->getTextValue(data->getValueVoidPtr(),index).c_str());

    /// should never happen....
    SP_PYERR_SETSTRING_OUTOFBOUND(0) ;
    return NULL;
}


static PyObject * Data_setValue(PyObject *self, PyObject * args)
{
    BaseData* data = get_basedata( self );
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo(); /// info about the data value
    int index;
    PyObject *value;

    if (!PyArg_ParseTuple(args, "iO", &index, &value)) {
        return NULL;
    }

    if ((unsigned int)index >= typeinfo->size())
    {
        SP_PYERR_SETSTRING_OUTOFBOUND(0);
        return NULL;
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

    /// should never happen....
    PyErr_BadArgument();
    return NULL;
}


static PyObject * Data_getValueTypeString(PyObject *self, PyObject* args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );
    return PyString_FromString(data->getValueTypeString().c_str());
}


static PyObject * Data_getValueString(PyObject *self, PyObject* args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );
    return PyString_FromString(data->getValueString().c_str());
}



//TODO(PR:304) a description of what this function is supposed to do?
static PyObject * Data_getSize(PyObject *self, PyObject * args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    int rowWidth = typeinfo->size();
    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();

    SP_MESSAGE_WARNING( "Data_getSize (this function always returns 0) rowWidth="<<rowWidth<<" nbRows="<<nbRows );

    //TODO(PR:304) So what ? with the WTF ? Do we remove it ?
    return PyInt_FromLong(0); //temp ==> WTF ?????
}


static PyObject * Data_setSize(PyObject *self, PyObject * args)
{
    BaseData* data = get_basedata( self );
    int size;
    if (!PyArg_ParseTuple(args, "i",&size))
    {
        return NULL;
    }
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    typeinfo->setSize((void*)data->getValueVoidPtr(),size);
    Py_RETURN_NONE;
}


static PyObject * Data_isSet(PyObject *self, PyObject* args)
{
    BaseData* data = get_basedata( self );

    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    return PyBool_FromLong(data->isSet());
}

static PyObject * Data_isPersistant(PyObject *self, PyObject * args)
{
    BaseData* data = get_basedata( self );

    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    return PyBool_FromLong(data->isPersistent());
}

static PyObject * Data_setPersistant(PyObject* self, PyObject* args)
{
    BaseData* data = get_basedata( self );

    PyObject* state = nullptr ;
    if (!PyArg_ParseTuple(args, "O", &state))
    {
        return NULL;
    }

    data->setPersistent(PyObject_IsTrue(state));

    Py_RETURN_NONE ;
}


static PyObject * Data_unset(PyObject *self, PyObject * /*args*/)
{
    BaseData* data = get_basedata( self );

    data->unset();

    Py_RETURN_NONE;
}


static PyObject * Data_updateIfDirty(PyObject *self, PyObject * args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );

    data->updateIfDirty();

    Py_RETURN_NONE;
}


static PyObject * Data_read(PyObject *self, PyObject * args)
{
    BaseData* data = get_basedata( self );

    PyObject *value;
    if (!PyArg_ParseTuple(args, "O",&value))
    {
        return NULL;
    }

    if (PyString_Check(value))
    {
        data->read(PyString_AsString(value));
    }
    else
    {
        PyErr_BadArgument();
        return NULL;
    }

    Py_RETURN_NONE;
}


static PyObject * Data_setParent(PyObject *self, PyObject * args)
{
    BaseData* data = get_basedata( self );

    PyObject *value;
    if (!PyArg_ParseTuple(args, "O",&value))
    {
        return NULL;
    }

    typedef PyPtr<BaseData> PyBaseData;

    if (PyString_Check(value))
    {
        data->setParent(PyString_AsString(value));
        data->setDirtyOutputs(); // forcing children updates (should it be done in BaseData?)
    }
    else if( dynamic_cast<BaseData*>(((PyBaseData*)value)->object) )
    {
        data->setParent( ((PyBaseData*)value)->object );
    }
    else
    {
        PyErr_BadArgument();
        return NULL;
    }

    Py_RETURN_NONE;
}

/// This function is actually returning the content of getLinkPath
//TODO(dmarchal 2017-08-02): This is awfull to have mismatch in behavior between python & C++
// with similar name.
static PyObject * Data_getParentPath(PyObject *self, PyObject * args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );
    return PyString_FromString(data->getLinkPath().c_str());
}

static PyObject * Data_hasParent(PyObject *self, PyObject * args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );

    return PyBool_FromLong( !data->getLinkPath().empty() ) ;
}


/// returns the complete link path name (i.e. following the shape "@/path/to/my/object.dataname")
static PyObject * Data_getLinkPath(PyObject * self, PyObject * /*args*/)
{
    BaseData* data = get_basedata( self );
    Base* owner = data->getOwner();

    if( owner )
    {
        if( BaseObject* obj = owner->toBaseObject() )
            return PyString_FromString(("@"+obj->getPathName()+"."+data->getName()).c_str());
        else if( BaseNode* node = owner->toBaseNode() )
            return PyString_FromString(("@"+node->getPathName()+"."+data->getName()).c_str());
    }

    /// default: no owner or owner of unknown type
    SP_MESSAGE_WARNING( "Data_getLinkName the Data has no known owner. Returning its own name." )
    return PyString_FromString(data->getName().c_str());
}


/// returns a pointer to the Data
static PyObject * Data_getValueVoidPtr(PyObject * self, PyObject * /*args*/)
{
    BaseData* data = get_basedata( self );

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    void* dataValueVoidPtr = const_cast<void*>(data->getValueVoidPtr()); /// data->beginEditVoidPtr();
    //TODO(PR:304) warning a endedit should be necessary somewhere (when releasing the python variable?)
    void* valueVoidPtr = typeinfo->getValuePtr(dataValueVoidPtr);

    /// N-dimensional arrays
    sofa::helper::vector<size_t> dimensions;
    dimensions.push_back( typeinfo->size(dataValueVoidPtr) );   /// total size to begin with
    const AbstractTypeInfo* valuetypeinfo = typeinfo;           /// to go trough encapsulated types (at the end, it will correspond to the finest type)

    while( valuetypeinfo->Container() )
    {
        size_t s = typeinfo->size();        /// the current type size
        dimensions.back() /= s;             /// to get the number of current type, the previous total size must be devided by the current type size
        dimensions.push_back( s );
        valuetypeinfo=valuetypeinfo->ValueType();
    }

    PyObject* shape = PyTuple_New(dimensions.size());
    for( size_t i=0; i<dimensions.size() ; ++i )
        PyTuple_SetItem( shape, i, PyLong_FromSsize_t( dimensions[i] ) );

    /// output = tuple( pointer, shape tuple, type name)
    PyObject* res = PyTuple_New(3);

    /// the data pointer
    PyTuple_SetItem( res, 0, PyLong_FromVoidPtr( valueVoidPtr ) );

    /// the shape
    PyTuple_SetItem( res, 1, shape );

    /// the most basic type name
    PyTuple_SetItem( res, 2, PyString_FromString( valuetypeinfo->name().c_str() ) );

    return res;
}


/// returns the number of times the Data was modified
static PyObject * Data_getCounter(PyObject * self, PyObject * args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );
    return PyInt_FromLong( data->getCounter() );
}


static PyObject * Data_isDirty(PyObject * self, PyObject * args)
{
    const size_t argSize = PyTuple_Size(args);
    if( argSize != 0 ) {
        PyErr_SetString(PyExc_RuntimeError, "This function does not accept any argument.") ;
        return NULL;
    }

    BaseData* data = get_basedata( self );
    return PyBool_FromLong( data->isDirty() );
}


/// implementation of __str__ to cast a Data to a string
static PyObject * Data_str(PyObject *self)
{
    BaseData* data = get_basedata( self );
    return PyString_FromString(data->getValueString().c_str());
}


static PyObject * Data_getAsACreateObjectParameter(PyObject * self, PyObject * args)
{
    return Data_getLinkPath(self, args);
}

static PyObject * Data_setValueString(PyObject *self, PyObject * args)
{
    return Data_read(self, args) ;
}



SP_CLASS_METHODS_BEGIN(Data)
SP_CLASS_METHOD(Data,getValueTypeString)
SP_CLASS_METHOD(Data,getValueString)
SP_CLASS_METHOD_DOC(Data,setValueString, "Set the value of the field from a string.")
SP_CLASS_METHOD(Data,setValue)
SP_CLASS_METHOD_DOC(Data,getValue, "Return the value at given index if the field is a vector.")
SP_CLASS_METHOD(Data,getSize)
SP_CLASS_METHOD(Data,setSize)
SP_CLASS_METHOD(Data,unset)
SP_CLASS_METHOD_DOC(Data,isSet, "Returns True/False if the data field has been setted.\n"
                                "if field.isSet():                                    \n"
                                "    print('set')                                     ")
SP_CLASS_METHOD_DOC(Data,isPersistant, "Returns True/False if the data field has the PERSISTANT(STORE) flag set/unset.")
SP_CLASS_METHOD_DOC(Data,setPersistant, "Change the PERSISTANT(STORE) flag of the data field. \n"
                                        "eg:                                                  \n"
                                        "    field.setPersistant(True)")
SP_CLASS_METHOD(Data,updateIfDirty)
SP_CLASS_METHOD(Data,read)
SP_CLASS_METHOD(Data,setParent)
SP_CLASS_METHOD_DOC(Data,getParentPath, "Returns the string containing the path to the field's parent. Return empty string if there is not parent")
SP_CLASS_METHOD_DOC(Data,hasParent, "Indicate if the string is linked to an other data field (its parent).")
SP_CLASS_METHOD(Data,getLinkPath)
SP_CLASS_METHOD(Data,getValueVoidPtr)
SP_CLASS_METHOD(Data,getCounter)
SP_CLASS_METHOD(Data,isDirty)
SP_CLASS_METHOD(Data,getAsACreateObjectParameter)
SP_CLASS_METHODS_END


SP_CLASS_ATTRS_BEGIN(Data)
SP_CLASS_ATTR(Data,name)
SP_CLASS_ATTR(Data,value)
SP_CLASS_ATTRS_END

namespace {
static struct patch {
    patch() {
        SP_SOFAPYTYPEOBJECT(Data).tp_str = Data_str; /// adding __str__ function
    }
} patcher;
}

SP_CLASS_TYPE_BASE_PTR_ATTR(Data, BaseData);
