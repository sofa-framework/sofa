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
#include <sofa/component/collision/InciseAlongPathPerformer.h>

#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{

helper::Creator<InteractionPerformer::InteractionPerformerFactory, InciseAlongPathPerformer>  InciseAlongPathPerformerClass("InciseAlongPath");


void InciseAlongPathPerformer::start()
{
    startBody=this->interactor->getBodyPicked();
    //std::cout << "clic: " << cpt << " => " << startBody.indexCollisionElement << std::endl;
    cpt++;
}

void InciseAlongPathPerformer::execute()
{
    //	  std::cout << "execute" << std::endl;
    if (currentMethod == 0) // incise from clic to clic
    {
        if (firstBody.body == NULL) // first clic
        {
            //std::cout << "First time" << std::endl;
            firstBody=startBody;
            //	      this->interactor->setBodyPicked(firstBody);
        }
        else
        {
            //	      startBody = this->interactor->getBodyPicked();
            if (firstBody.indexCollisionElement != startBody.indexCollisionElement)
            {
                std::cout << firstBody.indexCollisionElement << std::endl;
                std::cout << startBody.indexCollisionElement << std::endl;
                //std::cout << "Ecrit Second time" << std::endl;
                secondBody=startBody;
            }

        }



        //	    std::cout << "firstBody.point: " << firstBody.point << std::endl;
        if (firstBody.body == NULL || secondBody.body == NULL) return;
        //std::cout << "rentre" << std::endl;

        sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
        firstBody.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
        {
            //std::cerr << "Cutting from " << firstBody.point << " -------> " << secondBody.point << "\n";
            // core::componentmodel::topology::BaseMeshTopology::PointID point=
            topologyChangeManager.incisionCollisionModel(firstBody.body, firstBody.indexCollisionElement, firstBody.point,
                    secondBody.body,  secondBody.indexCollisionElement,  secondBody.point);
        }


        firstBody = secondBody;
        secondBody.body = NULL;

        this->interactor->setBodyPicked(secondBody);

    }
    else
    {

        BodyPicked currentBody=this->interactor->getBodyPicked();
        if (currentBody.body == NULL || startBody.body == NULL) return;

        if (currentBody.indexCollisionElement == startBody.indexCollisionElement) return;

        sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
        startBody.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
        {
            // std::cerr << "Cutting from " << startBody.point << " -------> " << currentBody.point << "\n";
            // core::componentmodel::topology::BaseMeshTopology::PointID point=
            topologyChangeManager.incisionCollisionModel(startBody.body, startBody.indexCollisionElement, startBody.point,
                    currentBody.body,  currentBody.indexCollisionElement,  currentBody.point);
        }
        startBody=currentBody;

        currentBody.body=NULL;
        this->interactor->setBodyPicked(currentBody);
    }

}



}
}
}

