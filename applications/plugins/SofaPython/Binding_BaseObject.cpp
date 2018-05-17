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

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::ObjectCreator ;

#include <sofa/core/behavior/BaseController.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/collision/CollisionAlgorithm.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/core/collision/Intersection.h>

#include "Binding_BaseObject.h"
#include "Binding_Base.h"
#include "PythonFactory.h"
#include "PythonToSofa.inl"

using sofa::core::objectmodel::BaseObject;

static BaseObject* get_baseobject(PyObject* self) {
    return sofa::py::unwrap<BaseObject>(self);
}


static PyObject * BaseObject_init(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->init();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_bwdInit(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->bwdInit();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_reinit(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->reinit();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_storeResetState(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->storeResetState();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_reset(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->reset();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_cleanup(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->cleanup();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_getContext(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    return sofa::PythonFactory::toPython(obj->getContext());
}

static PyObject * BaseObject_getMaster(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    return sofa::PythonFactory::toPython(obj->getMaster());
}


static PyObject * BaseObject_setSrc(PyObject *self, PyObject * args)
{
    BaseObject* obj = get_baseobject( self );
    char *valueString;
    PyObject *pyLoader;
    if (!PyArg_ParseTuple(args, "sO",&valueString,&pyLoader)) {
        return NULL;
    }
    BaseObject* loader = get_baseobject( self );
    obj->setSrc(valueString,loader);
    Py_RETURN_NONE;
}

static PyObject * BaseObject_getPathName(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    return PyString_FromString(obj->getPathName().c_str());
}

// the same as 'getPathName' with a extra prefix '@'
static PyObject * BaseObject_getLinkPath(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    return PyString_FromString(("@"+obj->getPathName()).c_str());
}


static PyObject * BaseObject_getSlaves(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    const BaseObject::VecSlaves& slaves = obj->getSlaves();

    PyObject *list = PyList_New(slaves.size());

    for (unsigned int i=0; i<slaves.size(); ++i)
        PyList_SetItem(list,i,sofa::PythonFactory::toPython(slaves[i].get()));

    return list;
}

static PyObject * BaseObject_getName(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    return PyString_FromString((obj->getName()).c_str());
}

extern "C" PyObject * BaseObject_getAsACreateObjectParameter(PyObject * self, PyObject *args)
{
    return BaseObject_getLinkPath(self, args);
}


static PyObject* BaseObject_getCategories(PyObject * self, PyObject *args)
{
    SOFA_UNUSED(args);
    BaseObject* obj = get_baseobject( self );

    if(!obj)
        return nullptr ;

    const sofa::core::objectmodel::BaseClass* mclass=obj->getClass();
    std::vector<std::string> categories;

    if (mclass->hasParent(sofa::core::objectmodel::ContextObject::GetClass()))
        categories.push_back("ContextObject");
    if (mclass->hasParent(sofa::core::visual::VisualModel::GetClass()))
        categories.push_back("VisualModel");
    if (mclass->hasParent(sofa::core::BehaviorModel::GetClass()))
        categories.push_back("BehaviorModel");
    if (mclass->hasParent(sofa::core::CollisionModel::GetClass()))
        categories.push_back("CollisionModel");
    if (mclass->hasParent(sofa::core::behavior::BaseMechanicalState::GetClass()))
        categories.push_back("MechanicalState");
    // A Mass is a technically a ForceField, but we don't want it to appear in the ForceField category
    if (mclass->hasParent(sofa::core::behavior::BaseForceField::GetClass()) && !mclass->hasParent(sofa::core::behavior::BaseMass::GetClass()))
        categories.push_back("ForceField");
    if (mclass->hasParent(sofa::core::behavior::BaseInteractionForceField::GetClass()))
        categories.push_back("InteractionForceField");
    if (mclass->hasParent(sofa::core::behavior::BaseProjectiveConstraintSet::GetClass()))
        categories.push_back("ProjectiveConstraintSet");
    if (mclass->hasParent(sofa::core::behavior::BaseConstraintSet::GetClass()))
        categories.push_back("ConstraintSet");
    if (mclass->hasParent(sofa::core::BaseMapping::GetClass()))
        categories.push_back("Mapping");
    if (mclass->hasParent(sofa::core::DataEngine::GetClass()))
        categories.push_back("Engine");
    if (mclass->hasParent(sofa::core::topology::TopologicalMapping::GetClass()))
        categories.push_back("TopologicalMapping");
    if (mclass->hasParent(sofa::core::behavior::BaseMass::GetClass()))
        categories.push_back("Mass");
    if (mclass->hasParent(sofa::core::behavior::OdeSolver::GetClass()))
        categories.push_back("OdeSolver");
    if (mclass->hasParent(sofa::core::behavior::ConstraintSolver::GetClass()))
        categories.push_back("ConstraintSolver");
    if (mclass->hasParent(sofa::core::behavior::BaseConstraintCorrection::GetClass()))
        categories.push_back("ConstraintSolver");
    if (mclass->hasParent(sofa::core::behavior::LinearSolver::GetClass()))
        categories.push_back("LinearSolver");
    if (mclass->hasParent(sofa::core::behavior::BaseAnimationLoop::GetClass()))
        categories.push_back("AnimationLoop");
    // Just like Mass and ForceField, we don't want TopologyObject to appear in the Topology category
    if (mclass->hasParent(sofa::core::topology::Topology::GetClass()) && !mclass->hasParent(sofa::core::topology::BaseTopologyObject::GetClass()))
        categories.push_back("Topology");
    if (mclass->hasParent(sofa::core::topology::BaseTopologyObject::GetClass()))
        categories.push_back("TopologyObject");
    if (mclass->hasParent(sofa::core::behavior::BaseController::GetClass()))
        categories.push_back("Controller");
    if (mclass->hasParent(sofa::core::loader::BaseLoader::GetClass()))
        categories.push_back("Loader");
    if (mclass->hasParent(sofa::core::collision::CollisionAlgorithm::GetClass()))
        categories.push_back("CollisionAlgorithm");
    if (mclass->hasParent(sofa::core::collision::Pipeline::GetClass()))
        categories.push_back("CollisionAlgorithm");
    if (mclass->hasParent(sofa::core::collision::Intersection::GetClass()))
        categories.push_back("CollisionAlgorithm");
    if (mclass->hasParent(sofa::core::objectmodel::ConfigurationSetting::GetClass()))
        categories.push_back("ConfigurationSetting");
    if (categories.empty())
        categories.push_back("Miscellaneous");

    PyObject *list = PyList_New(categories.size());
    for (unsigned int i=0; i<categories.size(); ++i)
        PyList_SetItem(list,i, PyString_FromString(categories[i].c_str())) ;

    return list ;
}

static PyObject * BaseObject_getTarget(PyObject *self, PyObject * args)
{
    SOFA_UNUSED(args);
    BaseObject* object = get_baseobject( self );

    if(!object)
        return nullptr ;

    /// Class description are stored in the factory creator.
    ObjectFactory::ClassEntry entry = ObjectFactory::getInstance()->getEntry(object->getClassName());
    if (!entry.creatorMap.empty())
    {
        ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(object->getTemplateName());
        if (it != entry.creatorMap.end() && *it->second->getTarget())
        {
            return PyString_FromString(it->second->getTarget()) ;
        }
    }

    return nullptr ;
}


SP_CLASS_METHODS_BEGIN(BaseObject)
SP_CLASS_METHOD(BaseObject,init)
SP_CLASS_METHOD(BaseObject,bwdInit)
SP_CLASS_METHOD(BaseObject,reinit)
SP_CLASS_METHOD(BaseObject,storeResetState)
SP_CLASS_METHOD(BaseObject,reset)
SP_CLASS_METHOD(BaseObject,cleanup)
SP_CLASS_METHOD(BaseObject,getContext)
SP_CLASS_METHOD(BaseObject,getMaster)
SP_CLASS_METHOD(BaseObject,setSrc)
SP_CLASS_METHOD(BaseObject,getPathName)
SP_CLASS_METHOD(BaseObject,getLinkPath)
SP_CLASS_METHOD(BaseObject,getSlaves)
SP_CLASS_METHOD(BaseObject,getName)
SP_CLASS_METHOD_DOC(BaseObject, getCategories,
                    "Returns a list of categories the current object belongs.")
SP_CLASS_METHOD_DOC(BaseObject, getTarget,
                    "Returns the target (plugin) that contains the current object.")
SP_CLASS_METHOD(BaseObject,getAsACreateObjectParameter)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(BaseObject,BaseObject,Base)
