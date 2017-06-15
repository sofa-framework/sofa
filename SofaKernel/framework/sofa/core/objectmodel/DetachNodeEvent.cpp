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
#include <sofa/core/objectmodel/DetachNodeEvent.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

SOFA_EVENT_CPP( DetachNodeEvent )

DetachNodeEvent::DetachNodeEvent(BaseNode* n)
    : node(n)
{
}

DetachNodeEvent::~DetachNodeEvent()
{
}

BaseNode* DetachNodeEvent::getNode() const
{
    return node;
}

bool DetachNodeEvent::contains(BaseNode* n) const
{
// Modified by FlorentF: A BaseNode is not always a Node from a Tree structure and can have more than one parent.
// This event should be specific to the GNodes and placed in modules and not framework
//     while (n != node && n != NULL)
//         n = n->getParent();
    return n == node;
}

bool DetachNodeEvent::contains(BaseObject* o) const
{
    return contains(o->getContext()->toBaseNode());
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
