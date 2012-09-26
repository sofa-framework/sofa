/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
using namespace sofa::simulation;
#include <sofa/core/ExecParams.h>
using namespace sofa::core;

#include "Binding_Node.h"
#include "Binding_Context.h"
#include "PythonVisitor.h"
#include "PythonScriptEvent.h"

extern "C" PyObject * Node_executeVisitor(PyObject *self, PyObject * args)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());

    PyObject* pyVisitor;
    if (!PyArg_ParseTuple(args, "O",&pyVisitor))
        return 0;
    PythonVisitor visitor(ExecParams::defaultInstance(),pyVisitor);
    node->executeVisitor(&visitor);

    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_getRoot(PyObject *self, PyObject * /*args*/)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());

    // BaseNode is not binded in SofaPython, so getRoot is binded in Node instead of BaseNode
    return SP_BUILD_PYSPTR(node->getRoot());
}

// step the simulation
extern "C" PyObject * Node_simulationStep(PyObject * self, PyObject * args)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    double dt;
    if (!PyArg_ParseTuple(args, "d",&dt))
        return 0;

    printf("Node_simulationStep node=%s dt=%f\n",node->getName().c_str(),(float)dt);

    getSimulation()->animate ( node, (SReal)dt );
//    simulation::getSimulation()->updateVisual( root );


    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_getChild(PyObject * self, PyObject * args)
{
    // BaseNode is not binded in SofaPython, so getChildNode is binded in Node instead of BaseNode
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
        return 0;
    if (!node || !path)
    {
        PyErr_BadArgument();
        return 0;
    }

    const objectmodel::BaseNode::Children& children = node->getChildren();
    Node *childNode = 0;
    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    for (unsigned int i=0; i<children.size(); ++i)
        if (children[i]->getName() == path)
        {
            childNode = dynamic_cast<Node*>(children[i]);
            break;
        }
    if (!childNode)
    {
        printf("<PYTHON> Error: Node.getChildNode(%s) not found.\n",path);
        return 0;
    }
    return SP_BUILD_PYSPTR(childNode);
}

extern "C" PyObject * Node_getChildren(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not binded in SofaPython, so getChildNode is binded in Node instead of BaseNode
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());

    const objectmodel::BaseNode::Children& children = node->getChildren();

    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    PyObject *list = PyList_New(children.size());

    for (unsigned int i=0; i<children.size(); ++i)
        PyList_SetItem(list,i,SP_BUILD_PYSPTR(children[i]));

    return list;
}

extern "C" PyObject * Node_getParents(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not binded in SofaPython, so getChildNode is binded in Node instead of BaseNode
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());

    const objectmodel::BaseNode::Children& parents = node->getParents();

    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    PyObject *list = PyList_New(parents.size());

    for (unsigned int i=0; i<parents.size(); ++i)
        PyList_SetItem(list,i,SP_BUILD_PYSPTR(parents[i]));

    return list;
}

extern "C" PyObject * Node_createChild(PyObject *self, PyObject * args)
{
    Node* obj=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    char *nodeName;
    if (!PyArg_ParseTuple(args, "s",&nodeName))
        return 0;
    return SP_BUILD_PYSPTR(obj->createChild(nodeName).get());
}

extern "C" PyObject * Node_addObject(PyObject *self, PyObject * args)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        return 0;
    BaseObject* object=dynamic_cast<BaseObject*>(((PySPtr<Base>*)pyChild)->object.get());
    if (!object)
    {
        PyErr_BadArgument();
        return 0;
    }
    node->addObject(object);

    //object->init();
    node->init(sofa::core::ExecParams::defaultInstance());

    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_removeObject(PyObject *self, PyObject * args)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        return 0;
    BaseObject* object=dynamic_cast<BaseObject*>(((PySPtr<Base>*)pyChild)->object.get());
    if (!object)
    {
        PyErr_BadArgument();
        return 0;
    }
    node->removeObject(object);
    node->init(sofa::core::ExecParams::defaultInstance());

    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_addChild(PyObject *self, PyObject * args)
{
    Node* obj=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        return 0;
    BaseNode* child=dynamic_cast<BaseNode*>(((PySPtr<Base>*)pyChild)->object.get());
    if (!child)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->addChild(child);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_removeChild(PyObject *self, PyObject * args)
{
    Node* obj=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        return 0;
    BaseNode* child=dynamic_cast<BaseNode*>(((PySPtr<Base>*)pyChild)->object.get());
    if (!child)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->removeChild(child);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_moveChild(PyObject *self, PyObject * args)
{
    Node* obj=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyChild;
    if (!PyArg_ParseTuple(args, "O",&pyChild))
        return 0;
    BaseNode* child=dynamic_cast<BaseNode*>(((PySPtr<Base>*)pyChild)->object.get());
    if (!child)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->moveChild(child);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_detachFromGraph(PyObject *self, PyObject * /*args*/)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    node->detachFromGraph();
    return Py_BuildValue("i",0);
}

extern "C" PyObject * Node_sendScriptEvent(PyObject *self, PyObject * args)
{
    Node* node=dynamic_cast<Node*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyUserData;
    char* eventName;
    if (!PyArg_ParseTuple(args, "sO",&eventName,&pyUserData))
    {
        PyErr_BadArgument();
        return 0;
    }
    PythonScriptEvent event(node->getName().c_str(),eventName,pyUserData);
    node->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
    return Py_BuildValue("i",0);
}


SP_CLASS_METHODS_BEGIN(Node)
SP_CLASS_METHOD(Node,executeVisitor)
SP_CLASS_METHOD(Node,getRoot)
SP_CLASS_METHOD(Node,simulationStep)
SP_CLASS_METHOD(Node,getChild)
SP_CLASS_METHOD(Node,getChildren)
SP_CLASS_METHOD(Node,getParents)
SP_CLASS_METHOD(Node,createChild)
SP_CLASS_METHOD(Node,addObject)
SP_CLASS_METHOD(Node,removeObject)
SP_CLASS_METHOD(Node,addChild)
SP_CLASS_METHOD(Node,removeChild)
SP_CLASS_METHOD(Node,moveChild)
SP_CLASS_METHOD(Node,detachFromGraph)
SP_CLASS_METHOD(Node,sendScriptEvent)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(Node,Node,Context)
