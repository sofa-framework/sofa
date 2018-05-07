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

#include "Binding_DataEngine.h"
#include "Binding_BaseObject.h"
#include "Binding_Base.h"
#include "PythonToSofa.inl"

using namespace sofa::core;
using namespace sofa::core::objectmodel;


static DataEngine* get_dataengine(PyObject* self) {
    return sofa::py::unwrap<DataEngine>(self);
}

static PyObject * DataEngine_updateIfDirty(PyObject *self, PyObject * /*args*/)
{
    DataEngine* engine = get_dataengine( self );;

    engine->updateIfDirty();

    Py_RETURN_NONE;
}

static PyObject * DataEngine_update(PyObject *self, PyObject * /*args*/)
{
    DataEngine* engine = get_dataengine( self );;

    engine->update();

    Py_RETURN_NONE;
}

static PyObject * DataEngine_isDirty(PyObject *self, PyObject * /*args*/)
{
    DataEngine* engine = get_dataengine( self );;

    return PyBool_FromLong( engine->isDirty() );
}


BaseData * helper_addNewIO(PyObject * self, PyObject * args, PyObject * kw)
{
    DataEngine* engine = get_dataengine( self );
    BaseData* NewData;

    NewData = helper_addNewData(args,kw,engine);

    if(NewData==nullptr)
    {
        msg_error("SofaPython") << "Adding new IO failed!";
        return nullptr;
    }
    NewData->setGroup(""); // Needs to be empty before it can be set to Input or Output ...

    return NewData;

}

static PyObject * DataEngine_addNewInput(PyObject *self, PyObject* args, PyObject * kw)
{
     DataEngine* engine = get_dataengine( self );

     BaseData * NewData = helper_addNewIO(self, args, kw);

     if (NewData == nullptr)
     {
         Py_RETURN_NONE;
     }

     // Check IO stuff
//     // TODO (sescaida 29.01.2018): maybe in the long term enforce that an Input can either be constant or only linked to an Output (for dat Simulink feelz)
//     BaseData* Parent = NewData->getParent();
//     char * ParentGroup;
//     if (Parent!=nullptr && strcmp(Parent->getGroup(), "Outputs")!=0)
//     {
//        msg_warning("SofaPython") << "Linking a Data defined as Input to a Data that is not an Output";
//     }

     engine->addInput(NewData);
     Py_RETURN_NONE;
}

static PyObject * DataEngine_addNewOutput(PyObject *self, PyObject* args, PyObject * kw)
{
    DataEngine* engine = get_dataengine( self );

    BaseData * NewData = helper_addNewIO(self,args, kw);

    if (NewData == nullptr)
    {
        Py_RETURN_NONE;
    }

    engine->addOutput(NewData);
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(DataEngine)
SP_CLASS_METHOD(DataEngine,updateIfDirty)
SP_CLASS_METHOD(DataEngine,update)
SP_CLASS_METHOD(DataEngine,isDirty)
SP_CLASS_METHOD_KW_DOC(DataEngine,addNewInput,
               "Creates a new sofa Data of the desired type and adds it as input to the PSDE-object. "
               )
SP_CLASS_METHOD_KW_DOC(DataEngine,addNewOutput,
               "Creates a new sofa Data of the desired type and adds it as output to the PSDE-object. "
               )
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(DataEngine,DataEngine,BaseObject)


