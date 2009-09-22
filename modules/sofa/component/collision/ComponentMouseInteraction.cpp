/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/collision/ComponentMouseInteraction.inl>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace collision
{
ComponentMouseInteraction::ComponentMouseInteraction():parentNode(NULL), nodeRayPick(NULL)/* ,mouseCollision(NULL) */
{
}

ComponentMouseInteraction::~ComponentMouseInteraction()
{
    nodeRayPick->execute< simulation::DeleteVisitor >();
    delete nodeRayPick;
}


void ComponentMouseInteraction::init(Node* node)
{
    parentNode = node;
    nodeRayPick = simulation::getSimulation()->newNode("RayPick");
}

void ComponentMouseInteraction::activate()
{
    parentNode->addChild(nodeRayPick);
    nodeRayPick->updateContext();
}

void ComponentMouseInteraction::deactivate()
{
    nodeRayPick->detachFromGraph();
}

void ComponentMouseInteraction::reset()
{
    mouseInteractor->cleanup();
}


#ifndef SOFA_DOUBLE
template class TComponentMouseInteraction<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class TComponentMouseInteraction<defaulttype::Vec3dTypes>;
#endif

#ifndef WIN32
#ifndef SOFA_DOUBLE
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Vec3fTypes> > ComponentMouseInteractionVec3fClass ("MouseSpringVec3f",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Vec3dTypes> > ComponentMouseInteractionVec3dClass ("MouseSpringVec3d",true);
#endif
#endif
}
}
}
