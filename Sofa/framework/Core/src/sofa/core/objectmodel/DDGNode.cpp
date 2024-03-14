/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/helper/BackTrace.h>
namespace sofa::core::objectmodel
{

/// Constructor
DDGNode::DDGNode()
{
}

DDGNode::~DDGNode()
{
    for(const auto it : inputs)
    {
        it->doDelOutput(this);
    }
    for(const auto it : outputs)
    {
        it->doDelInput(this);
    }
}

void DDGNode::setDirtyValue()
{
    bool& dirtyValue = dirtyFlags.dirtyValue;
    if (!dirtyValue)
    {
        dirtyValue = true;
        setDirtyOutputs();
    }
}

void DDGNode::setDirtyOutputs()
{
    bool& dirtyOutputs = dirtyFlags.dirtyOutputs;
    if (!dirtyOutputs)
    {
        dirtyOutputs = true;
        for(DDGLinkIterator it=outputs.begin(), itend=outputs.end(); it != itend; ++it)
        {
            (*it)->setDirtyValue();
        }
    }
}

void DDGNode::cleanDirty()
{
    bool& dirtyValue = dirtyFlags.dirtyValue;
    if (dirtyValue)
    {
        dirtyValue = false;
        cleanDirtyOutputsOfInputs();
    }
}

void DDGNode::notifyEndEdit()
{
    for(const auto it : outputs)
        it->notifyEndEdit();
}

void DDGNode::cleanDirtyOutputsOfInputs()
{
    for(const auto it : inputs)
        it->dirtyFlags.dirtyOutputs = false;
}

void DDGNode::addInput(DDGNode* n)
{
    if(std::find(inputs.begin(), inputs.end(), n) != inputs.end())
    {
        assert(false && "trying to add a DDGNode that is already in the input set.");
        return;
    }
    doAddInput(n);
    n->doAddOutput(this);
    setDirtyValue();
}

void DDGNode::delInput(DDGNode* n)
{
    /// It is not allowed to remove an entry that is not in the set.
    assert(std::find(inputs.begin(), inputs.end(), n) != inputs.end());

    doDelInput(n);
    n->doDelOutput(this);
}

void DDGNode::addOutput(DDGNode* n)
{
    if(std::find(outputs.begin(), outputs.end(), n) != outputs.end())
    {
        assert(false && "trying to add a DDGNode that is already in the output set.");
        return;
    }

    doAddOutput(n);
    n->doAddInput(this);
    n->setDirtyValue();
}

void DDGNode::delOutput(DDGNode* n)
{
    /// It is not allowed to remove an entry that is not in the set.
    assert(std::find(outputs.begin(), outputs.end(), n) != outputs.end());

    doDelOutput(n);
    n->doDelInput(this);
}

const DDGNode::DDGLinkContainer& DDGNode::getInputs()
{
    return inputs;
}

const DDGNode::DDGLinkContainer& DDGNode::getOutputs()
{
    return outputs;
}

void DDGNode::updateIfDirty() const
{
    if (isDirty())
    {
        const_cast <DDGNode*> (this)->update();
    }
}

void DDGNode::doAddInput(DDGNode* n)
{
    inputs.push_back(n);
}

void DDGNode::doDelInput(DDGNode* n)
{
    inputs.erase(std::remove(inputs.begin(), inputs.end(), n));
}

void DDGNode::doAddOutput(DDGNode* n)
{
    outputs.push_back(n);
}

void DDGNode::doDelOutput(DDGNode* n)
{
    outputs.erase(std::remove(outputs.begin(), outputs.end(), n));
}


} /// namespace sofa::core::objectmodel
