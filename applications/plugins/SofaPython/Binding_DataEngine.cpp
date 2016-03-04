/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "Binding_DataEngine.h"
#include "Binding_BaseObject.h"

#include <sofa/core/DataEngine.h>
using namespace sofa::core;
using namespace sofa::core::objectmodel;


extern "C" PyObject * DataEngine_updateIfDirty(PyObject *self, PyObject * /*args*/)
{
    DataEngine* engine=((PySPtr<Base>*)self)->object->toDataEngine();

    engine->updateIfDirty();

    Py_RETURN_NONE;
}

extern "C" PyObject * DataEngine_update(PyObject *self, PyObject * /*args*/)
{
    DataEngine* engine=((PySPtr<Base>*)self)->object->toDataEngine();

    engine->update();

    Py_RETURN_NONE;
}

extern "C" PyObject * DataEngine_isDirty(PyObject *self, PyObject * /*args*/)
{
    DataEngine* engine=((PySPtr<Base>*)self)->object->toDataEngine();

    return PyBool_FromLong( engine->isDirty() );
}

SP_CLASS_METHODS_BEGIN(DataEngine)
SP_CLASS_METHOD(DataEngine,updateIfDirty)
SP_CLASS_METHOD(DataEngine,update)
SP_CLASS_METHOD(DataEngine,isDirty)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(DataEngine,DataEngine,BaseObject)


