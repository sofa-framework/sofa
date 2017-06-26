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
#include "PythonMacros.h"
#include <sofa/simulation/Simulation.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/VisualVisitor.h>
using namespace sofa::simulation;
#include <sofa/core/ExecParams.h>
using namespace sofa::core;
using namespace sofa::core::objectmodel;
#include "Binding_Node.h"
#include "Binding_Context.h"
#include "PythonVisitor.h"
#include "PythonScriptEvent.h"
#include "PythonFactory.h"

extern "C" PyObject * Node_executeVisitor(PyObject *self, PyObject * args)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    PyObject* pyVisitor;
    if (!PyArg_ParseTuple(args, "O",&pyVisitor))
        Py_RETURN_NONE;
    PythonVisitor visitor(ExecParams::defaultInstance(),pyVisitor);
    node->executeVisitor(&visitor);

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_getRoot(PyObject *self, PyObject * /*args*/)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    // BaseNode is not bound in SofaPython, so getRoot is bound in Node instead of BaseNode
    return sofa::PythonFactory::toPython(node->getRoot());
}

// step the simulation
extern "C" PyObject * Node_simulationStep(PyObject * self, PyObject * args)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    double dt;
    if (!PyArg_ParseTuple(args, "d",&dt))
        Py_RETURN_NONE;

//    printf("Node_simulationStep node=%s dt=%f\n",node->getName().c_str(),(float)dt);

    getSimulation()->animate ( node, (SReal)dt );
//    simulation::getSimulation()->updateVisual( root );


    Py_RETURN_NONE;
}

// reset a node
extern "C" PyObject * Node_reset(PyObject * self, PyObject * /*args*/)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    getSimulation()->reset(node);

    Py_RETURN_NONE;
}

// init a node
extern "C" PyObject * Node_init(PyObject * self, PyObject * /*args*/)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    getSimulation()->init(node);

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_getChild(PyObject * self, PyObject * args, PyObject * kw)
{
    // BaseNode is not bound in SofaPython, so getChildNode is bound in Node instead of BaseNode
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    char *path;

    if (!PyArg_ParseTuple(args, "s",&path))
        Py_RETURN_NONE;
    if (!node || !path)
    {
        PyErr_BadArgument();
        return NULL;
    }

    bool warning = true;
    if (kw && PyDict_Size(kw)>0)
    {
        PyObject* keys = PyDict_Keys(kw);
        PyObject* values = PyDict_Values(kw);
        for (int i=0; i<PyDict_Size(kw); i++)
        {
            PyObject *key = PyList_GetItem(keys,i);
            PyObject *value = PyList_GetItem(values,i);
            if( !strcmp(PyString_AsString(key),"warning") )
            {
                if PyBool_Check(value)
                    warning = (value==Py_True);
                break;
            }
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    const objectmodel::BaseNode::Children& children = node->getChildren();
    Node *childNode = 0;
    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    for (unsigned int i=0; i<children.size(); ++i)
        if (children[i]->getName() == path)
        {
            childNode = down_cast<Node>(children[i]);
            break;
        }
    if (!childNode)
    {
        if( warning ) SP_MESSAGE_ERROR( "Node.getChild(\""<<path<<"\") not found.")
        Py_RETURN_NONE;
    }
    return sofa::PythonFactory::toPython(childNode);
}

extern "C" PyObject * Node_getChildren(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not bound in SofaPython, so getChildNode is bound in Node instead of BaseNode
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    const objectmodel::BaseNode::Children& children = node->getChildren();

    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    PyObject *list = PyList_New(children.size());

    for (unsigned int i=0; i<children.size(); ++i)
        PyList_SetItem(list,i,sofa::PythonFactory::toPython(children[i]));

    return list;
}

extern "C" PyObject * Node_getParents(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not bound in SofaPython, so getChildNode is bound in Node instead of BaseNode
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    const objectmodel::BaseNode::Children& parents = node->getParents();

    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    PyObject *list = PyList_New(parents.size());

    for (unsigned int i=0; i<parents.size(); ++i)
        PyList_SetItem(list,i,sofa::PythonFactory::toPython(parents[i]));

    return list;
}

extern "C" PyObject * Node_getPathName(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not bound in SofaPython, so getPathName is bound in Node instead
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    return PyString_FromString(node->getPathName().c_str());
}

extern "C" PyObject * Node_getRootPath(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not bound in SofaPython, so getRootPath is bound in Node instead
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    return PyString_FromString(node->getRootPath().c_str());
}

// the same as 'getPathName' with a extra prefix '@'
extern "C" PyObject * Node_getLinkPath(PyObject * self, PyObject * /*args*/)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    return PyString_FromString(("@"+node->getPathName()).c_str());
}

extern "C" PyObject * Node_createChild(PyObject *self, PyObject * args)
{
    Node* obj=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    char *nodeName;
    if (!PyArg_ParseTuple(args, "s",&nodeName))
        Py_RETURN_NONE;
    Node* child = obj->createChild(nodeName).get();
    return sofa::PythonFactory::toPython(child);
}

extern "C" PyObject * Node_addObject_Impl(PyObject *self, PyObject * args, PyObject * kw, bool printWarnings)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        Py_RETURN_NONE;

    bool warning = printWarnings;
    if (kw && PyDict_Size(kw)>0)
    {
        PyObject* keys = PyDict_Keys(kw);
        PyObject* values = PyDict_Values(kw);
        for (int i=0; i<PyDict_Size(kw); i++)
        {
            PyObject *key = PyList_GetItem(keys,i);
            PyObject *value = PyList_GetItem(values,i);
            if( !strcmp(PyString_AsString(key),"warning") )
            {
                if PyBool_Check(value)
                    warning = (value==Py_True);
                break;
            }
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    BaseObject* object=((PySPtr<Base>*)pyChild)->object->toBaseObject();
    if (!object)
    {
        PyErr_BadArgument();
        return NULL;
    }
    node->addObject(object);

    if (warning && node->isInitialized())
        SP_MESSAGE_WARNING( "Sofa.Node.addObject called on a node("<<node->getName()<<") that is already initialized ("<<object->getName()<<")" )

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_addObject(PyObject * self, PyObject * args, PyObject * kw)
{
    return Node_addObject_Impl( self, args, kw, true );
}
extern "C" PyObject * Node_addObject_noWarning(PyObject * self, PyObject * args)
{
    SP_MESSAGE_DEPRECATED("Node_addObject_noWarning is deprecated, use the keyword warning=False in Node_addObject instead.")
    return Node_addObject_Impl( self, args, NULL, false );
}

extern "C" PyObject * Node_removeObject(PyObject *self, PyObject * args)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        Py_RETURN_NONE;
    BaseObject* object=((PySPtr<Base>*)pyChild)->object->toBaseObject();
    if (!object)
    {
        PyErr_BadArgument();
        return NULL;
    }
    node->removeObject(object);

    // no init, if you need to init, you can call it yourself!
//    node->init(sofa::core::ExecParams::defaultInstance());

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_addChild(PyObject *self, PyObject * args)
{
    Node* obj=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        Py_RETURN_NONE;
    BaseNode* child=((PySPtr<Base>*)pyChild)->object->toBaseNode();
    if (!child)
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->addChild(child);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_removeChild(PyObject *self, PyObject * args)
{
    Node* obj=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        Py_RETURN_NONE;
    BaseNode* child=((PySPtr<Base>*)pyChild)->object->toBaseNode();
    if (!child)
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->removeChild(child);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_moveChild(PyObject *self, PyObject * args)
{
    Node* obj=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        Py_RETURN_NONE;
    BaseNode* child=((PySPtr<Base>*)pyChild)->object->toBaseNode();
    if (!child)
    {
        PyErr_BadArgument();
        return NULL;
    }
    obj->moveChild(child);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_detachFromGraph(PyObject *self, PyObject * /*args*/)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    node->detachFromGraph();
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_sendScriptEvent(PyObject *self, PyObject * args)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    PyObject* pyUserData;
    char* eventName;
    if (!PyArg_ParseTuple(args, "sO",&eventName,&pyUserData))
    {
        PyErr_BadArgument();
        return NULL;
    }
    PythonScriptEvent event(node,eventName,pyUserData);
    node->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_sendKeypressedEvent(PyObject *self, PyObject * args)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    char* eventName;
    if (!PyArg_ParseTuple(args, "s",&eventName))
    {
        PyErr_BadArgument();
        return NULL;
    }
    sofa::core::objectmodel::KeypressedEvent event(eventName ? eventName[0] : '\0');
    down_cast<Node>(node->getRoot())->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_sendKeyreleasedEvent(PyObject *self, PyObject * args)
{
    Node* node=down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    char* eventName;
    if (!PyArg_ParseTuple(args, "s",&eventName))
    {
        PyErr_BadArgument();
        return NULL;
    }
    sofa::core::objectmodel::KeyreleasedEvent event(eventName ? eventName[0] : '\0');
    down_cast<Node>(node->getRoot())->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_getMechanicalState(PyObject * self, PyObject * /*args*/)
{
    Node* node = down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    behavior::BaseMechanicalState* state = node->mechanicalState.get();

    if( state ) return sofa::PythonFactory::toPython(state);

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_getMechanicalMapping(PyObject * self, PyObject * /*args*/)
{
    Node* node = down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    sofa::core::BaseMapping* mapping = node->mechanicalMapping.get();

    if( mapping ) return sofa::PythonFactory::toPython(mapping);

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_propagatePositionAndVelocity(PyObject * self, PyObject * /*args*/)
{
    Node* node = down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());

    node->execute<MechanicalPropagatePositionAndVelocityVisitor>(sofa::core::MechanicalParams::defaultInstance()); // only mechanical mappings
    node->execute<UpdateMappingVisitor>(sofa::core::MechanicalParams::defaultInstance()); // propagating position and velocity through non mechanical mappings
    node->execute<VisualUpdateVisitor>(sofa::core::MechanicalParams::defaultInstance());

    Py_RETURN_NONE;
}

extern "C" PyObject * Node_isInitialized(PyObject *self, PyObject * /*args*/)
{
    Node* node = down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    return PyBool_FromLong( node->isInitialized() );
}

extern "C" PyObject * Node_printGraph(PyObject *self, PyObject * /*args*/)
{
    Node* node = down_cast<Node>(((PySPtr<Base>*)self)->object->toBaseNode());
    getSimulation()->print(node);
    Py_RETURN_NONE;
}

extern "C" PyObject * Node_getAsACreateObjectParameter(PyObject * self, PyObject *args)
{
    return Node_getLinkPath(self, args);
}

SP_CLASS_METHODS_BEGIN(Node)
SP_CLASS_METHOD(Node,executeVisitor)
SP_CLASS_METHOD(Node,getRoot)
SP_CLASS_METHOD(Node,simulationStep)
SP_CLASS_METHOD(Node,reset)
SP_CLASS_METHOD(Node,init)
SP_CLASS_METHOD_KW(Node,getChild)
SP_CLASS_METHOD(Node,getChildren)
SP_CLASS_METHOD(Node,getParents)
SP_CLASS_METHOD(Node,getPathName)
SP_CLASS_METHOD(Node,getRootPath)
SP_CLASS_METHOD(Node,getLinkPath)
SP_CLASS_METHOD(Node,createChild)
SP_CLASS_METHOD_KW(Node,addObject)
SP_CLASS_METHOD(Node,addObject_noWarning) // deprecated
SP_CLASS_METHOD(Node,removeObject)
SP_CLASS_METHOD(Node,addChild)
SP_CLASS_METHOD(Node,removeChild)
SP_CLASS_METHOD(Node,moveChild)
SP_CLASS_METHOD(Node,detachFromGraph)
SP_CLASS_METHOD(Node,sendScriptEvent)
SP_CLASS_METHOD(Node,sendKeypressedEvent)
SP_CLASS_METHOD(Node,sendKeyreleasedEvent)
SP_CLASS_METHOD(Node,getMechanicalState)
SP_CLASS_METHOD(Node,getMechanicalMapping)
SP_CLASS_METHOD(Node,propagatePositionAndVelocity)
SP_CLASS_METHOD(Node,isInitialized)
SP_CLASS_METHOD(Node,printGraph)
SP_CLASS_METHOD(Node,getAsACreateObjectParameter)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(Node,Node,Context)
