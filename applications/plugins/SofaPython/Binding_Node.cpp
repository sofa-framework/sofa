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

extern "C" PyObject * Node_getChildNode(PyObject * self, PyObject * args)
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

SP_CLASS_METHODS_BEGIN(Node)
SP_CLASS_METHOD(Node,executeVisitor)
SP_CLASS_METHOD(Node,getRoot)
SP_CLASS_METHOD(Node,simulationStep)
SP_CLASS_METHOD(Node,getChildNode)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(Node,Node,Context)
