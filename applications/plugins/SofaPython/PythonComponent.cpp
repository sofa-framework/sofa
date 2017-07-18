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
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/******************************************************************************
*  Contributors:                                                              *
*  - damien.marchal@univ-lille1.fr                                            *
******************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::Base ;

#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/IdleEvent.h>
using sofa::core::objectmodel::IdleEvent ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::RegisterObject ;


#include "PythonMacros.h"
#include "Binding_BaseObject.h"

#include "PythonFactory.h"
using sofa::PythonFactory ;

#include "PythonComponent.h"


namespace sofa
{

namespace component
{

namespace _pythoncomponent_
{
using sofa::core::objectmodel::BaseNode ;
using sofa::simulation::Node ;

PythonComponent::PythonComponent() : BaseObject(),
    m_source(initData(&m_source, std::string(""), "template", "Current template source" , true, false))
{
    m_source.setGroup("Python.properties");
}

PythonComponent::~PythonComponent(){}


SOFA_DECL_CLASS(PythonComponent)
int PythonComponentClass = core::RegisterObject("An object template encoded as parsed hson-py object.")
        .add< PythonComponent >();


} // namespace __pythoncomponent_

} // namespace component

} // namespace sofa

using sofa::core::objectmodel::BaseData ;
using sofa::component::python::PythonComponent ;

static PyObject * PythonComponent_setSource(PyObject *self, PyObject * args)
{
    PythonComponent* obj= dynamic_cast<PythonComponent*>(((PySPtr<Base>*)self)->object.get()) ;
    if(obj->m_rawPython)
        Py_DECREF(obj->m_rawPython);

    obj->m_rawPython = nullptr ;
    if (!PyArg_ParseTuple(args, "O", &(obj->m_rawPython))) {
        return NULL;
    }

    std::stringstream s ;
    PyObject* tmpstr = PyObject_Repr(obj->m_rawPython);
    s << PyString_AsString(tmpstr) ;
    obj->m_source.setValue(s.str()) ;

    Py_DECREF(tmpstr);
    Py_INCREF(obj->m_rawPython);
    Py_INCREF(obj->m_rawPython);
    return obj->m_rawPython ;
}

static PyObject * PythonComponent_getSource(PyObject *self, PyObject * args)
{
    SOFA_UNUSED(args);
    PythonComponent* obj= dynamic_cast<PythonComponent*>(((PySPtr<Base>*)self)->object.get()) ;
    if(obj->m_rawPython){
        Py_INCREF(obj->m_rawPython);
        return obj->m_rawPython ;
    }
    return nullptr ;
}


SP_CLASS_METHODS_BEGIN(PythonComponent)
SP_CLASS_METHOD(PythonComponent, setSource)
SP_CLASS_METHOD(PythonComponent, getSource)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(PythonComponent,PythonComponent,BaseObject)


