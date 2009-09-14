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
#define SOFA_COMPONENT_COLLISION_MOUSEINTERACTOR_CPP
#include <sofa/component/collision/MouseInteractor.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(MouseInteractor)

int MouseInteractorClass = core::RegisterObject("Perform tasks related to the interaction with the mouse")
        .add< MouseInteractor<defaulttype::Vec3Types> >();


template class SOFA_COMPONENT_COLLISION_API MouseInteractor<defaulttype::Vec3Types>;




void BaseMouseInteractor::updatePosition( double )
{

    if (isRemovingElement && collisionModel)
    {
        core::CollisionElementIterator collisionElement( collisionModel, indexCollisionElement);

        sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
        collisionModel->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier) topologyChangeManager.removeItemsFromCollisionModel(collisionElement);
        collisionModel=NULL;
        isRemovingElement=false;
    }
    else if (isIncising)
    {

        sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
        elementsPicked[0].body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
        {
            std::cerr << "Cutting from " << elementsPicked[0].point << " -------> " << elementsPicked[1].point << "\n";
            // core::componentmodel::topology::BaseMeshTopology::PointID point=
            topologyChangeManager.incisionCollisionModel(elementsPicked[0].body, elementsPicked[0].indexCollisionElement, elementsPicked[0].point,
                    elementsPicked[1].body, elementsPicked[1].indexCollisionElement, elementsPicked[1].point);
        }
        isIncising=false;
    }
}



void BaseMouseInteractor::draw()
{
    if (collisionModel)
    {
        if (isAttached)
            glColor4f(1.0f,0.0f,0.0f,1.0f);
        else
            glColor4f(0.0f,1.0f,0.0f,1.0f);

        glDisable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3);
        collisionModel->draw(indexCollisionElement);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


        glColor4f(1,1,1,1);
        glLineWidth(1);
    }
}
}
}
}
