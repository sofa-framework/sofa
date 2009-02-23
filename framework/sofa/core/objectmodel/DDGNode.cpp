/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/DDGNode.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

DDGNode::~DDGNode()
{
    for(std::list< DDGNode* >::iterator it=inputs.begin(); it!=inputs.end(); ++it)
        (*it)->outputs.remove(this);
    for(std::list< DDGNode* >::iterator it=outputs.begin(); it!=outputs.end(); ++it)
        (*it)->inputs.remove(this);
}

void DDGNode::setDirty()
{
    if (!dirty)
    {
        dirty = true;
        for(std::list<DDGNode*>::iterator it=outputs.begin(); it!=outputs.end(); ++it)
        {
            (*it)->setDirty();
        }
    }
}

void DDGNode::cleanDirty()
{
    dirty = false;
}

bool DDGNode::isDirty()
{
    return dirty;
}

void DDGNode::addInput(DDGNode* n)
{
    inputs.push_back(n);
    n->outputs.push_back(this);
}

void DDGNode::delInput(DDGNode* n)
{
    inputs.remove(n);
    n->outputs.remove(this);
}

void DDGNode::addOutput(DDGNode* n)
{
    outputs.push_back(n);
    n->inputs.push_back(this);
}

void DDGNode::delOutput(DDGNode* n)
{
    outputs.remove(n);
    n->inputs.remove(this);
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
