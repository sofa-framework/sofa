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

    if (cpt == 0) // register first position of incision
    {
        firstIncisionBody = startBody;
        cpt++;
        initialNbTriangles = startBody.body->getMeshTopology()->getNbTriangles();
    }
}

void InciseAlongPathPerformer::execute()
{

    if (freezePerformer) // This performer has been freezed
    {
        startBody=this->interactor->getBodyPicked();
        return;
    }

    if (currentMethod == 0) // incise from clic to clic
    {
        if (firstBody.body == NULL) // first clic
            firstBody=startBody;
        else
        {
            if (firstBody.indexCollisionElement != startBody.indexCollisionElement)
                secondBody=startBody;
        }


        if (firstBody.body == NULL || secondBody.body == NULL) return;

        sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
        firstBody.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
        {
            topologyChangeManager.incisionCollisionModel(firstBody.body, firstBody.indexCollisionElement, firstBody.point,
                    secondBody.body,  secondBody.indexCollisionElement,  secondBody.point,
                    snapingValue, snapingBorderValue );
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
            topologyChangeManager.incisionCollisionModel(startBody.body, startBody.indexCollisionElement, startBody.point,
                    currentBody.body,  currentBody.indexCollisionElement,  currentBody.point,
                    snapingValue, snapingBorderValue );
        }
        startBody = currentBody;
        firstBody = currentBody;

        currentBody.body=NULL;
        this->interactor->setBodyPicked(currentBody);
    }

}

void InciseAlongPathPerformer::setPerformerFreeze()
{
    freezePerformer = true;
    if (fullcut)
        this->PerformCompleteIncision();

    fullcut = true;
}

void InciseAlongPathPerformer::PerformCompleteIncision()
{
    if (firstIncisionBody.body == NULL || startBody.body == NULL) return;

    if (firstIncisionBody.indexCollisionElement == startBody.indexCollisionElement) return;

    sofa::core::componentmodel::topology::TopologyModifier* topologyModifier;
    startBody.body->getContext()->get(topologyModifier);

    // Handle Removing of topological element (from any type of topology)
    if(topologyModifier)
    {
        topologyChangeManager.incisionCollisionModel(firstIncisionBody.body, initialNbTriangles, firstIncisionBody.point,
                startBody.body,  startBody.indexCollisionElement,  startBody.point,
                snapingValue, snapingBorderValue );
    }

    startBody = firstIncisionBody;
    firstIncisionBody.body = NULL;

    finishIncision = false; //Incure no second cut
}

InciseAlongPathPerformer::~InciseAlongPathPerformer()
{
    if (secondBody.body)
        secondBody.body= NULL;

    if (firstBody.body)
        firstBody.body = NULL;

    if (startBody.body)
        startBody.body = NULL;

    if (firstIncisionBody.body)
        firstIncisionBody.body = NULL;

    this->interactor->setBodyPicked(firstIncisionBody);
}



}
}
}

