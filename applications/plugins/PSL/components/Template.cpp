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


#include <SofaPython/PythonMacros.h>
#include <SofaPython/Binding_BaseObject.h>

#include <SofaPython/PythonFactory.h>
using sofa::PythonFactory ;

#include "Template.h"
using sofa::core::objectmodel::Data ;

#include "SofaPython/PythonEnvironment.h"
using sofa::simulation::PythonEnvironment ;

#include "SofaPython/Binding_Base.h"
#include "SofaPython/PythonToSofa.inl"
using sofa::core::objectmodel::BaseData ;
using sofa::component::_template_::Template ;



namespace sofa
{

namespace component
{

namespace _template_
{
using sofa::core::objectmodel::BaseNode ;
using sofa::simulation::Node ;

Template::Template() : BaseObject(),
    m_template(initData(&m_template, std::string(""), "psl_source", "Current template source" , true, false))
{
    m_template.setGroup("PSL");
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
    PythonEnvironment::gil lock();

    std::map<Base*, Base*> updateList ;
    for(BaseData* data : m_trackedDatas)
    {
        if(m_dataTracker.isDirty(*data))
        {
            if(data->getName()=="psl_source") {
                std::cout << "Template re-instanciation..." << data->getValueString() << std::endl ;

                Base* base = data->getOwner() ;
                Node* node = dynamic_cast<Node*>(base) ;

                /// Re-instantiate it.
                PyObject* pDict = PyModule_GetDict(PyImport_AddModule("pslengine"));
                PyObject* pFunc = PyDict_GetItemString(pDict, "reinstanciateAllTemplates");

                if(!pDict || !pFunc)
                {
                    std::cout << "UNABLE TO GET FUNCTIOn " << pFunc << std::endl ;
                    return;
                }

                if (PyCallable_Check(pFunc))
                {
                    PyObject* tgt = PythonFactory::toPython(data->getOwner());
                    PyObject* res = PyObject_CallFunction(pFunc, "O", tgt) ;

                    if(PyErr_Occurred())
                        PyErr_Print();
                }
            }
            else
            {
                updateList[data->getOwner()] = data->getOwner() ;
            }
        }

    }
    m_dataTracker.clean();

    for( auto& kv : updateList )
    {
        std::cout << "C++ update of Template" << kv.first->getName() << std::endl ;

        /// Re-instantiate it.
        PyObject* pDict = PyModule_GetDict(PyImport_AddModule("pslengine"));
        PyObject* pFunc = PyDict_GetItemString(pDict, "reinstanciateATemplateInstance");

        if(!pDict || !pFunc)
        {
            std::cout << "UNABLE TO GET FUNCTIOn " << pFunc << std::endl ;
            return;
        }

        if (PyCallable_Check(pFunc))
        {
            Node* targetInstance = dynamic_cast<Node*>(kv.first) ;
            PyObject* src = PythonFactory::toPython(this);
            PyObject* tgt = PythonFactory::toPython(targetInstance);
            PyObject* res = PyObject_CallFunction(pFunc, "OO", tgt, src) ;

            if(PyErr_Occurred())
                PyErr_Print();
        }

    }
}

void Template::addDataToTrack(BaseData* d)
{
    m_dataTracker.trackData(*d) ;
    m_trackedDatas.push_back(d);
}

SOFA_DECL_CLASS(Template)
int TemplateClass = core::RegisterObject("An object template encoded as parsed hson-py object.")
        .add< Template >();


} // namespace _baseprefab_

} // namespace component

} // namespace sofa


static PyObject * Template_setTemplate(PyObject *self, PyObject * args)
{
    PythonEnvironment::gil lock();

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
    SOFA_UNUSED(args);
    PythonEnvironment::gil lock();

    Template* obj= dynamic_cast<Template*>(((PySPtr<Base>*)self)->object.get()) ;
    if(obj->m_rawTemplate){
        Py_INCREF(obj->m_rawTemplate);
        return obj->m_rawTemplate ;
    }
    return nullptr ;
}

static PyObject * Template_trackData(PyObject *self, PyObject * args)
{
    PythonEnvironment::gil lock();

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


static Base* get_base(PyObject* self) {
    return sofa::py::unwrap<Base>(self);
}


//TODO(dmarchal 2017-07-15) Factor that before PR.
char* getStringCopy(char *c)
{
    char* tmp = new char[strlen(c)+1] ;
    strcpy(tmp,c);
    return tmp ;
}

static PyObject * Template_createATrackedData(PyObject *self, PyObject *args ) {
    PythonEnvironment::gil lock();

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
        bd = t;
    }
    else if(dataRawType[0] == 'b'){
        Data<bool>* t = new Data<bool>();
        bd = t;
    }
    else if(dataRawType[0] == 'd'){
        Data<int>* t = new Data<int>();
        bd = t;
    }
    else if(dataRawType[0] == 'f'){
        Data<float>* t = new Data<float>();
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
    bd->setName(dataName) ;
    bd->setHelp(dataHelp);
    bd->read( tmp.str() ) ;
    bd->setGroup(dataClass);

    return SP_BUILD_PYPTR(Data,BaseData,bd,false);
}



SP_CLASS_METHODS_BEGIN(Template)
SP_CLASS_METHOD(Template, setTemplate)
SP_CLASS_METHOD(Template, getTemplate)
SP_CLASS_METHOD(Template, trackData)
SP_CLASS_METHOD(Template, createATrackedData)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(Template,Template,BaseObject)


