/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "Binding_Base.h"
#include "Binding_Data.h"
#include "Binding_Link.h"

#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
using namespace sofa::core::objectmodel;

#include <sofa/helper/logging/Messaging.h>

#include "PythonFactory.h"
#include "PythonToSofa.inl"

#include "PythonEnvironment.h"
using sofa::simulation::PythonEnvironment ;

static Base* get_base(PyObject* self) {
    return sofa::py::unwrap<Base>(self);
}

//TODO(dmarchal 2017-07-15) Factor that before PR.
/// This function converts an PyObject into a sofa string.
/// string that can be safely parsed in helper::vector<int> or helper::vector<double>
static std::ostream& pythonToSofaDataString(PyObject* value, std::ostream& out)
{
    /// String are just returned as string.
    if (PyString_Check(value))
    {
        return out << PyString_AsString(value) ;
    }


    if( PySequence_Check(value) )
    {
        /// It is a sequence...so we can iterate over it.
        PyObject *iterator = PyObject_GetIter(value);
        if(iterator)
        {
            bool first = true;
            while(PyObject* next = PyIter_Next(iterator))
            {
                if(first) first = false;
                else out << ' ';

                pythonToSofaDataString(next, out);
                Py_DECREF(next);
            }
            Py_DECREF(iterator);

            if (PyErr_Occurred())
            {
                msg_error("SofaPython") << "error while iterating." << msgendl
                                        << PythonEnvironment::getStackAsString() ;
            }
            return out;
        }
    }


    /// Check if the object has an explicit conversion to a Sofa path. If this is the case
    /// we use it.
    if( PyObject_HasAttrString(value, "getAsACreateObjectParameter") ){
       PyObject* retvalue = PyObject_CallMethod(value, (char*)"getAsACreateObjectParameter", nullptr) ;
       return pythonToSofaDataString(retvalue, out);
    }

    /// Default conversion for standard type:
    if( !(PyInt_Check(value) || PyLong_Check(value) || PyFloat_Check(value) || PyBool_Check(value) ))
    {
        msg_warning("SofaPython") << "You are trying to convert a non primitive type to Sofa using the 'str' operator." << msgendl
                                  << "Automatic conversion is provided for: String, Integer, Long, Float and Bool and Sequences." << msgendl
                                  << "Other objects should implement the method getAsACreateObjectParameter(). " << msgendl
                                  << "This function should return a string usable as a parameter in createObject()." << msgendl
                                  << "So to remove this message you must add a method getAsCreateObjectParameter(self) "
                                     "to the object you are passing the createObject function." << msgendl
                                  << PythonEnvironment::getStackAsString() ;
    }


    PyObject* tmpstr=PyObject_Str(value);
    out << PyString_AsString(tmpstr) ;
    Py_DECREF(tmpstr) ;
    return out ;
}

//TODO(dmarchal 2017-07-15) Factor that before PR.
char* getStringCopy(char *c)
{
    char* tmp = new char[strlen(c)+1] ;
    strcpy(tmp,c);
    return tmp ;
}

static PyObject * Base_realAddData(PyObject *self, PyObject *args )
{
    Base* obj = get_base(self);
    PyObject* pydata;

    if (!PyArg_ParseTuple(args, "O", &pydata)) {
        return NULL;
    }

    BaseData* data = sofa::py::unwrap<BaseData>(pydata);
    obj->addData(data) ;

    Py_RETURN_NONE;
}

static PyObject * Base_addData(PyObject *self, PyObject *args ) {
    Base* obj = get_base(self);
    char* dataName;
    char* dataClass;
    char* dataHelp;
    char* dataRawType;
    PyObject* dataValue;

    if (!PyArg_ParseTuple(args, "ssssO", &dataName, &dataClass, &dataHelp, &dataRawType, &dataValue)) {
        return NULL;
    }

    dataName = getStringCopy(dataName) ;
    dataClass = getStringCopy(dataClass) ;
    dataHelp  = getStringCopy(dataHelp) ;


    //TODO(dmarchal 2017-07-15) il y a une fuite mémoire ici. A cause de l'init qui ne fait
    // pas de copie des chaines... mais juste du swallow du coup on se retrouve à copier les nom
    // à chaque template. C'est méga naze !
    BaseData* bd = nullptr ;
    if(dataRawType[0] == 's'){
        Data<std::string>* t = new Data<std::string>() ;
        t = new(t) Data<std::string>(obj->initData(t, std::string(""), dataName, dataHelp)) ;
        bd = t;
    }
    else if(dataRawType[0] == 'b'){
        Data<bool>* t = new Data<bool>();
        t = new(t) Data<bool>(obj->initData(t, true, dataName, dataHelp)) ;
        bd = t;
    }
    else if(dataRawType[0] == 'd'){
        Data<int>* t = new Data<int>();
        t = new (t) Data<int> (obj->initData(t, 0, dataName, dataHelp)) ;
        bd = t;
    }
    else if(dataRawType[0] == 'f'){
        Data<float>* t = new Data<float>();
        t = new (t) Data<float>(obj->initData(t, 0.0f, dataName, dataHelp)) ;
        bd = t;
    }
    else{
        std::stringstream msg;
        msg << "Invalid data type '" << dataRawType << "'. Supported type are: s(tring), d(ecimal), f(float), b(oolean)" ;
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return NULL;
    }

    std::stringstream tmp;
    pythonToSofaDataString(dataValue, tmp) ;
    bd->read( tmp.str() ) ;
    bd->setGroup(dataClass);

    Py_RETURN_NONE ;
}

static PyObject * Base_getData(PyObject *self, PyObject *args ) {
    Base* obj = get_base(self);
    char *dataName;
    if (!PyArg_ParseTuple(args, "s", &dataName)) {
        return NULL;
    }

    BaseData * data = obj->findData(dataName);
    if (!data)
    {
        Py_RETURN_NONE ;
    }

    return SP_BUILD_PYPTR(Data,BaseData,data,false);
}

static PyObject * Base_getLink(PyObject *self, PyObject *args ) {
    Base* obj = get_base(self);
    char *dataName;
    if (!PyArg_ParseTuple(args, "s", &dataName)) {
        return NULL;
    }

    BaseLink *link = obj->findLink(dataName);
    if (!link)
    {
        Py_RETURN_NONE ;
    }

    return SP_BUILD_PYPTR(Link,BaseLink,link,false);
}

static PyObject * Base_findData(PyObject *self, PyObject *args ) {
    Base* obj = get_base(self);
    char *dataName;

    if (!PyArg_ParseTuple(args, "s", &dataName)) {
        return NULL;
    }

    BaseData * data = obj->findData(dataName);
    if (!data) {
        std::stringstream tmp ;
        if( obj->hasField(dataName) ) {
            tmp <<"object '"<<obj->getName()<<"' has a field '"<<dataName<<"' but it is not a Data";
        } else {
            tmp << "object '"<<obj->getName()<<"' does no have a field '"<<dataName<<"'";
            obj->writeDatas(tmp,";");
        }

        PyErr_SetString(PyExc_RuntimeError, tmp.str().c_str());
        return NULL;
    }

    /// special cases... from factory (e.g DisplayFlags, OptionsGroup)
    {
        PyObject* res = sofa::PythonFactory::toPython(data);
        if( res ) return res;
    }

    return SP_BUILD_PYPTR(Data,BaseData,data,false);
}


static PyObject * Base_findLink(PyObject *self, PyObject *args) {
    Base* obj = get_base(self);
    char *linkName;
    if (!PyArg_ParseTuple(args, "s", &linkName)) {
        return NULL;
    }

    BaseLink * link = obj->findLink(linkName);
    if (!link) {
        std::stringstream tmp ;
        if( obj->hasField(linkName) ) {
            tmp << "object '"<<obj->getName()<<"' has a field '"<<linkName<<"' but it is not a Link";
        } else {
            tmp <<"object '"<<obj->getName()<<"' does no have a field '"<<linkName<<"'" << msgendl;
            obj->writeDatas(tmp,";");
        }

        PyErr_SetString(PyExc_RuntimeError, tmp.str().c_str());
        return NULL;
    }

    return SP_BUILD_PYPTR(Link,BaseLink,link,false);
}

/// Generic accessor to Data fields (in python native type)
static PyObject* Base_GetAttr(PyObject *o, PyObject *attr_name) {
    Base* obj = get_base(o);
    char *attrName = PyString_AsString(attr_name);

    /// see if a Data field has this name...
    if( BaseData * data = obj->findData(attrName) ) {
        /// special cases... from factory (e.g DisplayFlags, OptionsGroup)
        if( PyObject* res = sofa::PythonFactory::toPython(data) ) {
            return res;
        } else {
            /// the data type is not known by the factory, let's create the right Python type....
            return GetDataValuePython(data);
        }
    }

    /// see if a Link has this name...
    if( BaseLink * link = obj->findLink(attrName) ) {
        /// we have our link... let's create the right Python type....
        return GetLinkValuePython(link);
    }

    return PyObject_GenericGetAttr(o,attr_name);
}

static int Base_SetAttr(PyObject *o, PyObject *attr_name, PyObject *v) {
    /// attribute does not exist: see if a Data field has this name...
    Base* obj = get_base(o);
    char *attrName = PyString_AsString(attr_name);

    if (BaseData * data = obj->findData(attrName)) {
        /// data types in Factory can have a specific setter
        if( PyObject* pyData = sofa::PythonFactory::toPython(data) ) {
            return PyObject_SetAttrString( pyData, "value", v );
        } else {
            /// the data type is not known by the factory, let's use the default implementation
            return SetDataValuePython(data,v);
        }
    }

    if (BaseLink * link = obj->findLink(attrName)) {
        return SetLinkValuePython(link,v);
    }

    return PyObject_GenericSetAttr(o,attr_name,v);
}

static PyObject * Base_getClassName(PyObject * self, PyObject * /*args*/) {
    /// BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
    Base* node = get_base(self);

    return PyString_FromString(node->getClassName().c_str());
}

static PyObject * Base_getTemplateName(PyObject * self, PyObject * /*args*/) {
    /// BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
    Base* node = get_base(self);

    return PyString_FromString(node->getTemplateName().c_str());
}

static PyObject * Base_getName(PyObject * self, PyObject * /*args*/) {
    /// BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
    Base* node = get_base(self);

    return PyString_FromString(node->getName().c_str());
}

static PyObject * Base_getDataFields(PyObject *self, PyObject * /*args*/) {
    Base * component = get_base(self);

    const sofa::helper::vector<BaseData*> dataFields = component->getDataFields();

    PyObject * pyDict = PyDict_New();
    for (size_t i=0; i<dataFields.size(); i++) {
        PyDict_SetItem(pyDict, PyString_FromString(dataFields[i]->getName().c_str()),
                       GetDataValuePython(dataFields[i]));
    }

    return pyDict;
}


/// This function is named this way because someone give the getDataFields to the one
/// that returns a dictionary of (name, value) which is not coherente with the c++
/// name of the function.
/// If you are a brave hacker, courageous enough to break backward compatibility you can
/// probably fix all this mess.
static PyObject * Base_getListOfDataFields(PyObject *self, PyObject * /*args*/) {
    Base * component = get_base(self);

    const sofa::helper::vector<BaseData*> dataFields = component->getDataFields();

    PyObject * pyList = PyList_New(dataFields.size());
    for (unsigned int i = 0; i < dataFields.size(); ++i) {
        PyList_SetItem(pyList, i, SP_BUILD_PYPTR(Data,BaseData,dataFields[i],false)) ;
    }

    return pyList;
}

static PyObject * Base_getListOfLinks(PyObject *self, PyObject * /*args*/) {
    Base * component = get_base(self);

    const sofa::helper::vector<BaseLink*> links = component->getLinks() ;

    PyObject * pyList = PyList_New(links.size());
    for (unsigned int i = 0; i < links.size(); ++i) {
        PyList_SetItem(pyList, i, SP_BUILD_PYPTR(Link, BaseLink, links[i], false)) ;
    }

    return pyList;
}

/// down cast to the lowest type known by the factory
/// there is maybe a more pythonish way to do so? :)
static PyObject * Base_downCast(PyObject *self, PyObject * /*args*/) {
    Base* component = get_base(self);
    return sofa::PythonFactory::toPython(component);
}

SP_CLASS_METHODS_BEGIN(Base)
SP_CLASS_METHOD(Base,addData)
SP_CLASS_METHOD(Base,realAddData)
SP_CLASS_METHOD(Base,findData)
SP_CLASS_METHOD(Base,findLink)
SP_CLASS_METHOD(Base,getData)
SP_CLASS_METHOD(Base,getLink)
SP_CLASS_METHOD(Base,getClassName)
SP_CLASS_METHOD(Base,getTemplateName)
SP_CLASS_METHOD(Base,getName)
SP_CLASS_METHOD(Base,getDataFields)
SP_CLASS_METHOD(Base,getListOfDataFields)
SP_CLASS_METHOD(Base,getListOfLinks)
SP_CLASS_METHOD(Base,downCast)
SP_CLASS_METHODS_END;


SP_CLASS_ATTRS_BEGIN(Base)
SP_CLASS_ATTRS_END;

SP_CLASS_TYPE_BASE_SPTR_ATTR_GETATTR(Base, Base)
