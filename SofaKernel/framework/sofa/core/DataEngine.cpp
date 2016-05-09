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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/core/DataEngine.h>

namespace sofa
{

namespace core
{

DataEngine::DataEngine()
{
    addLink(&(this->core::objectmodel::DDGNode::inputs));
    addLink(&(this->core::objectmodel::DDGNode::outputs));
}

DataEngine::~DataEngine()
{
}

void DataEngine::updateAllInputsIfDirty()
{
    const DDGLinkContainer& inputs = DDGNode::getInputs();
    for(size_t i=0, iend=inputs.size() ; i<iend ; ++i )
    {
        static_cast<core::objectmodel::BaseData*>(inputs[i])->updateIfDirty();
    }
}

/// Add a new input to this engine
void DataEngine::addInput(objectmodel::BaseData* n)
{
    if (n->getOwner() == this && (!n->getGroup() || !n->getGroup()[0]))
        n->setGroup("Inputs"); // set the group of input Datas if not yet set
    core::objectmodel::DDGNode::addInput(n);
}

/// Remove an input from this engine
void DataEngine::delInput(objectmodel::BaseData* n)
{
    core::objectmodel::DDGNode::delInput(n);
}

/// Add a new output to this engine
void DataEngine::addOutput(objectmodel::BaseData* n)
{
    if (n->getOwner() == this && (!n->getGroup() || !n->getGroup()[0]))
        n->setGroup("Outputs"); // set the group of output Datas if not yet set
    core::objectmodel::DDGNode::addOutput(n);
}

/// Remove an output from this engine
void DataEngine::delOutput(objectmodel::BaseData* n)
{
    core::objectmodel::DDGNode::delOutput(n);
}

} // namespace core

} // namespace sofa
