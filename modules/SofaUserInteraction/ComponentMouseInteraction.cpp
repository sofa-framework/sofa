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
#define SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_CPP
#include <SofaUserInteraction/ComponentMouseInteraction.inl>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/Factory.inl>


using namespace sofa::simulation;

namespace sofa
{

namespace component
{

namespace collision
{


ComponentMouseInteraction::ComponentMouseInteraction():
    nodeRayPick(NULL),
    mouseInSofa(NULL),
    mouseInteractor(NULL)
{
}

ComponentMouseInteraction::~ComponentMouseInteraction()
{
    if (nodeRayPick)
    {
        nodeRayPick->execute< simulation::DeleteVisitor >(sofa::core::ExecParams::defaultInstance());
        nodeRayPick.reset();
    }
}



void ComponentMouseInteraction::attach(Node* parentNode)
{
    if(parentNode)
    {
        if (!nodeRayPick)
        {
            nodeRayPick = parentNode->createChild("MouseInteraction");
            createInteractionComponents(parentNode,nodeRayPick.get());
        }
        else
        {
            parentNode->addChild(nodeRayPick);
        }
    }
}

void ComponentMouseInteraction::detach()
{
    if (nodeRayPick)
        nodeRayPick->detachFromGraph();
}

void ComponentMouseInteraction::reset()
{
    if (mouseInteractor)
        mouseInteractor->cleanup();
}

#ifndef SOFA_DOUBLE
template class TComponentMouseInteraction<defaulttype::Vec2fTypes>;
template class TComponentMouseInteraction<defaulttype::Vec3fTypes>;
template class TComponentMouseInteraction<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class TComponentMouseInteraction<defaulttype::Vec2dTypes>;
template class TComponentMouseInteraction<defaulttype::Vec3dTypes>;
template class TComponentMouseInteraction<defaulttype::Rigid3dTypes>;
#endif

#ifndef SOFA_DOUBLE
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Vec2fTypes> > ComponentMouseInteractionVec2fClass ("MouseSpringVec2f",true);
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Vec3fTypes> > ComponentMouseInteractionVec3fClass ("MouseSpringVec3f",true);
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Rigid3fTypes> > ComponentMouseInteractionRigid3fClass ("MouseSpringRigid3f",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Vec2dTypes> > ComponentMouseInteractionVec2dClass ("MouseSpringVec2d",true);
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Vec3dTypes> > ComponentMouseInteractionVec3dClass ("MouseSpringVec3d",true);
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<defaulttype::Rigid3dTypes> > ComponentMouseInteractionRigid3dClass ("MouseSpringRigid3d",true);
#endif

}
}
namespace helper
{
template class SOFA_USER_INTERACTION_API Factory<std::string, component::collision::ComponentMouseInteraction, core::objectmodel::BaseContext*>;
}
}
