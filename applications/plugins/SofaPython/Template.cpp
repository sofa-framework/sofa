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

#include "Template.h"


namespace sofa
{

namespace component
{

namespace _template_
{
using sofa::core::objectmodel::BaseNode ;
using sofa::simulation::Node ;

Template::Template() : BaseObject(),
    m_template(initData(&m_template, std::string(""), "template", "Current template source" , true, false))
{
    m_template.setGroup("Template.properties");
}

Template::~Template(){}

void Template::handleEvent(Event *event)
{
    if (dynamic_cast<sofa::core::objectmodel::IdleEvent *>(event))
        checkAndDoUpdates() ;
    else
        BaseObject::handleEvent(event);
}

void Template::checkAndDoUpdates()
{
    for(BaseData* data : m_trackedDatas){
        if(m_dataTracker.isDirty(*data)){
            std::cout << "Data is cleaned ... or not " << data->getValueString() << std::endl ;
            m_dataTracker.clean(*data) ;


            Base* base = data->getOwner() ;
            Node* node = dynamic_cast<Node*>(base) ;

            //std::vector<BaseObject*> ct;
            //node->getTreeObjects<BaseObject, std::vector<BaseObject*>>(&ct);

            //for(auto& i : ct){
            //    node->removeObject(i) ;
            //}

            /// Re-instantiate it.
            PyObject* pDict = PyModule_GetDict(PyImport_AddModule("pysonloader"));
            PyObject* pFunc = PyDict_GetItemString(pDict, "reinstanciateTemplate");

            if(!pDict || !pFunc)
                return;

            std::cout << "INSTANTIATE TEMPLATE " << pFunc << std::endl ;

            if (PyCallable_Check(pFunc))
            {
                PyObject* tgt = PythonFactory::toPython(data->getOwner());
                PyObject* res = PyObject_CallFunction(pFunc, "O", tgt) ;

                if(PyErr_Occurred())
                    PyErr_Print();
            }

        }
    }
}

void Template::addDataToTrack(BaseData* d)
{
    std::cout << "ADDING TO DATA TRACKER.. " << d->getName() <<  std::endl ;
    m_dataTracker.trackData(*d) ;
    m_trackedDatas.push_back(d);
}

SOFA_DECL_CLASS(Template)
int TemplateClass = core::RegisterObject("An object template encoded as parsed hson-py object.")
        .add< Template >();


} // namespace _baseprefab_

} // namespace component

} // namespace sofa

using sofa::core::objectmodel::BaseData ;
using sofa::component::_template_::Template ;

static PyObject * Template_setTemplate(PyObject *self, PyObject * args)
{
    Template* obj= dynamic_cast<Template*>(((PySPtr<Base>*)self)->object.get()) ;
    if(obj->m_rawTemplate)
        Py_DECREF(obj->m_rawTemplate);

    obj->m_rawTemplate = nullptr ;
    if (!PyArg_ParseTuple(args, "O", &(obj->m_rawTemplate))) {
        return NULL;
    }

    std::stringstream s ;
    PyObject* tmpstr = PyObject_Repr(obj->m_rawTemplate);
    s << PyString_AsString(tmpstr) ;
    obj->m_template.setValue(s.str()) ;

    Py_DECREF(tmpstr);
    Py_INCREF(obj->m_rawTemplate);
    Py_INCREF(obj->m_rawTemplate);
    return obj->m_rawTemplate ;
}

static PyObject * Template_getTemplate(PyObject *self, PyObject * args)
{
    Template* obj= dynamic_cast<Template*>(((PySPtr<Base>*)self)->object.get()) ;
    if(obj->m_rawTemplate){
        Py_INCREF(obj->m_rawTemplate);
        return obj->m_rawTemplate ;
    }
    return nullptr ;
}

static PyObject * Template_trackData(PyObject *self, PyObject * args)
{

    Template* obj = dynamic_cast<Template*>(((PySPtr<Base>*)self)->object.get()) ;
    PyObject* o  {nullptr} ;
    if (!PyArg_ParseTuple(args, "O", &o)) {
        return NULL ;
    }

    BaseData* bd = ((PyPtr<BaseData>*)o)->object ;
    if(obj && bd)
        obj->addDataToTrack(bd) ;
    else
        return NULL ;
    Py_RETURN_NONE ;
}


SP_CLASS_METHODS_BEGIN(Template)
SP_CLASS_METHOD(Template, setTemplate)
SP_CLASS_METHOD(Template, getTemplate)
SP_CLASS_METHOD(Template, trackData)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(Template,Template,BaseObject)


