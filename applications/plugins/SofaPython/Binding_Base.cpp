/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

// stop copypasting for the lord's sake
static Base* get_base(PyObject* obj) {
    return ((PySPtr<Base>*)obj)->object.get();
}

static PyObject * Base_findData(PyObject *self, PyObject *args ) {
    Base* obj = get_base(self);
    char *dataName;
    
    if (!PyArg_ParseTuple(args, "s", &dataName)) {
        return NULL;
    }
    
    BaseData * data = obj->findData(dataName);
    if (!data) {
        if( obj->hasField(dataName) ) {
            msg_error("Base_findData")<<"object '"<<obj->getName()<<"' has a field '"<<dataName<<"' but it is not a Data"; } else {
            msg_error("Base_findData")<<"object '"<<obj->getName()<<"' does no have a field '"<<dataName<<"'";
            std::stringstream s;
            obj->writeDatas(s,";");
            msg_error("Base_findData")<<s.str();
        }

        PyErr_BadArgument();
        return NULL;
    }



    // special cases... from factory (e.g DisplayFlags, OptionsGroup)
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
        if( obj->hasField(linkName) ) {
            msg_error("Base_findLink")<<"object '"<<obj->getName()<<"' has a field '"<<linkName<<"' but it is not a Link";
        } else {
            msg_error("Base_findLink")<<"object '"<<obj->getName()<<"' does no have a field '"<<linkName<<"'";
            std::stringstream s;
            obj->writeDatas(s,";");
            msg_error("Base_findLink")<<s.str();
        }

        PyErr_BadArgument();
        return NULL;
    }

    return SP_BUILD_PYPTR(Link,BaseLink,link,false);
}


// Generic accessor to Data fields (in python native type)
static PyObject* Base_GetAttr(PyObject *o, PyObject *attr_name) {
    Base* obj = get_base(o);
    char *attrName = PyString_AsString(attr_name);
//    printf("Base_GetAttr type=%s name=%s attrName=%s\n",obj->getClassName().c_str(),obj->getName().c_str(),attrName);

    // see if a Data field has this name...
    if( BaseData * data = obj->findData(attrName) ) {
        // special cases... from factory (e.g DisplayFlags, OptionsGroup)
        if( PyObject* res = sofa::PythonFactory::toPython(data) ) {
            return res;
        } else {// the data type is not known by the factory, let's create the right Python type....
            return GetDataValuePython(data);
        }
    }

    // see if a Link has this name...
    if( BaseLink * link = obj->findLink(attrName) ) {
        return GetLinkValuePython(link); // we have our link... let's create the right Python type....
    }
    
    //        printf("Base_GetAttr ERROR data not found - type=%s name=%s attrName=%s\n",obj->getClassName().c_str(),obj->getName().c_str(),attrName);
    return PyObject_GenericGetAttr(o,attr_name);
}

static int Base_SetAttr(PyObject *o, PyObject *attr_name, PyObject *v) {
    // attribute does not exist: see if a Data field has this name...
    Base* obj = get_base(o);
    char *attrName = PyString_AsString(attr_name);

//    printf("Base_SetAttr name=%s\n",attrName);
    if (BaseData * data = obj->findData(attrName)) {
        // data types in Factory can have a specific setter
        if( PyObject* pyData = sofa::PythonFactory::toPython(data) ) {
            return PyObject_SetAttrString( pyData, "value", v );
        } else {// the data type is not known by the factory, let's use the default implementation
            return SetDataValuePython(data,v);
        }
    }

    if (BaseLink * link = obj->findLink(attrName)) {
        return SetLinkValuePython(link,v);
    }
    
    return PyObject_GenericSetAttr(o,attr_name,v);
}

static PyObject * Base_getClassName(PyObject * self, PyObject * /*args*/) {
    // BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
    Base* node = get_base(self);

    return PyString_FromString(node->getClassName().c_str());
}

static PyObject * Base_getTemplateName(PyObject * self, PyObject * /*args*/) {
    // BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
    Base* node = get_base(self);

    return PyString_FromString(node->getTemplateName().c_str());
}

static PyObject * Base_getName(PyObject * self, PyObject * /*args*/) {
    // BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
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

SP_CLASS_METHODS_BEGIN(Base)
SP_CLASS_METHOD(Base,findData)
SP_CLASS_METHOD(Base,findLink)
SP_CLASS_METHOD(Base,getClassName)
SP_CLASS_METHOD(Base,getTemplateName)
SP_CLASS_METHOD(Base,getName)
SP_CLASS_METHOD(Base,getDataFields)
SP_CLASS_METHODS_END

//SP_CLASS_DATA_ATTRIBUTE(Base,name)

SP_CLASS_ATTRS_BEGIN(Base)
//SP_CLASS_ATTR(Base,name)
SP_CLASS_ATTRS_END;

SP_CLASS_TYPE_BASE_SPTR_ATTR_GETATTR(Base, Base)
