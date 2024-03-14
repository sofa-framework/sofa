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
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/objectmodel/BaseNode.h>

#include <cstdlib>
#include <cmath>

namespace sofa::core::behavior
{

BaseAnimationLoop::BaseAnimationLoop()
    : l_node(initLink("targetNode","Link to the scene's node that will be processed by the loop"))
    , m_resetTime(0.)
    , d_computeBoundingBox(initData(&d_computeBoundingBox, !SOFA_NO_UPDATE_BBOX, "computeBoundingBox", "If true, compute the global bounding box of the scene at each time step. Used mostly for rendering."))
{}

BaseAnimationLoop::~BaseAnimationLoop()
{}

void BaseAnimationLoop::init()
{
    if(!l_node)
        l_node = dynamic_cast<sofa::core::objectmodel::BaseNode*>(getContext());
}

void BaseAnimationLoop::storeResetState()
{
    const objectmodel::BaseContext * c = this->getContext();

    if (c != nullptr)
        m_resetTime = c->getTime();
}

SReal BaseAnimationLoop::getResetTime() const
{
    return m_resetTime;
}

bool BaseAnimationLoop::insertInNode( objectmodel::BaseNode* node )
{
    node->addAnimationLoop(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseAnimationLoop::removeInNode( objectmodel::BaseNode* node )
{
    node->removeAnimationLoop(this);
    Inherit1::removeInNode(node);
    return true;
}

} // namespace sofa::core::behavior

