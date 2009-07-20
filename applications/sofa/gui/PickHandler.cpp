/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/PickHandler.h>

#include <sofa/component/collision/RayContact.h>

#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>

#include <sofa/helper/Factory.inl>
#include <sofa/component/collision/ComponentMouseInteraction.inl>

#include <iostream>



namespace sofa
{

namespace gui
{

PickHandler::PickHandler():interactorInUse(false), mouseStatus(DEACTIVATED)
{
    operations[0] = operations[1] = operations[2] = NULL;

    mouseNode = simulation::getSimulation()->newNode("Mouse");

    mouseContainer = new MouseContainer; mouseContainer->resize(1);
    mouseContainer->setName("MousePosition");
    mouseNode->addObject(mouseContainer);


    mouseCollision = new MouseCollisionModel;
    mouseCollision->setNbRay(1);
    mouseCollision->getRay(0).l() = 1000000;
    mouseCollision->setName("MouseCollisionModel");
    mouseNode->addObject(mouseCollision);


    mouseNode->init();
    mouseContainer->init();
    mouseCollision->init();




    typedef component::collision::ComponentMouseInteraction::ComponentMouseInteractionFactory MouseFactory;
    const MouseFactory *factory = MouseFactory::getInstance();
    for (MouseFactory::const_iterator it = factory->begin(); it != factory->end(); ++it)
    {
        instanceComponents.push_back(it->second->createInstance(NULL));
        instanceComponents.back()->init(mouseNode);
    }
    interaction = instanceComponents.back();
}

PickHandler::~PickHandler()
{
//       for (unsigned int i=0;i<instanceComponents.size();++i) delete instanceComponents[i];
}

void PickHandler::reset()
{
    if (!interactorInUse)
    {
        interaction->reset();
    }
}
void PickHandler::activateRay(bool act)
{
    if (interactorInUse && !act)
    {
        mouseNode->detachFromGraph();

        interaction->deactivate();
        interactorInUse=false;

        elementsPicked[0] = BodyPicked();
        elementsPicked[1] = BodyPicked();
    }
    else if (!interactorInUse && act)
    {
        Node *root = static_cast<Node*>(simulation::getSimulation()->getContext());
        root->addChild(mouseNode);
        interaction->activate();
        interactorInUse=true;
    }
}




bool PickHandler::needToCastRay()
{
    return !getInteraction()->mouseInteractor->isMouseAttached();
}


void PickHandler::setCompatibleInteractor()
{

    if (!lastPicked.body) return;
    if (interaction->isCompatible(lastPicked.body->getContext())) return;
    for (unsigned int i=0; i<instanceComponents.size(); ++i)
    {
        if (instanceComponents[i] != interaction &&
            instanceComponents[i]->isCompatible(lastPicked.body->getContext()))
        {
            interaction->deactivate();
            interaction = instanceComponents[i];
            interaction->activate();
        }
    }

}


void PickHandler::updateRay(const sofa::defaulttype::Vector3 &position,const sofa::defaulttype::Vector3 &orientation)
{
    if (!interactorInUse) return;

    mouseCollision->getRay(0).origin() = position+orientation*interaction->mouseInteractor->getDistanceFromMouse();
    mouseCollision->getRay(0).direction() = orientation;

    if (needToCastRay())
    {
        lastPicked=findCollision();
        setCompatibleInteractor();

        interaction->mouseInteractor->setCollisionElement(lastPicked.body, lastPicked.indexCollisionElement);
    }


    switch (mouseStatus)
    {
    case PRESSED:
    {
        operations[mouseButton]->start();
        mouseStatus=ACTIVATED;
        break;
    }
    case RELEASED:
    {
        operations[mouseButton]->end();
        mouseStatus=DEACTIVATED;
        break;
    }
    case ACTIVATED:
    {
        operations[mouseButton]->execution();
    }
    case DEACTIVATED:
    {
        //Nothing to do
    }
    }
}

//Clear the node create, and destroy all its components
void PickHandler::handleMouseEvent(MOUSE_STATUS status, MOUSE_BUTTON button)
{
    if (!interaction) return;

    mouseButton=button;
    mouseStatus=status;
}


ComponentMouseInteraction *PickHandler::getInteraction()
{
    return interaction;
}


component::collision::BodyPicked PickHandler::findCollision()
{
    const defaulttype::Vector3& origin          = mouseCollision->getRay(0).origin();
    const defaulttype::Vector3& direction       = mouseCollision->getRay(0).direction();
    const double& maxLength                     = mouseCollision->getRay(0).l();


    BodyPicked result;
    const std::set< sofa::component::collision::BaseRayContact*> &contacts = mouseCollision->getContacts();
    for (std::set< sofa::component::collision::BaseRayContact*>::const_iterator it=contacts.begin(); it != contacts.end(); ++it)
    {

        const sofa::helper::vector<core::componentmodel::collision::DetectionOutput*>& output = (*it)->getDetectionOutputs();
        sofa::core::CollisionModel *modelInCollision;
        for (unsigned int i=0; i<output.size(); ++i)
        {

            if (output[i]->elem.first.getCollisionModel() == mouseCollision)
            {
                modelInCollision = output[i]->elem.second.getCollisionModel();
                if (!modelInCollision->isSimulated()) continue;


                const double d = (output[i]->point[1]-origin)*direction;
                if (d<0.0 || d>maxLength) continue;
                if (result.body == NULL || d < result.rayLength)
                {
                    result.body=modelInCollision;
                    result.indexCollisionElement = output[i]->elem.second.getIndex();
                    result.point = output[i]->point[1];
                    result.dist  = (output[i]->point[1]-output[i]->point[0]).norm();
                    result.rayLength  = d;
                }
            }
            else if (output[i]->elem.second.getCollisionModel() == mouseCollision)
            {
                modelInCollision = output[i]->elem.first.getCollisionModel();
                if (!modelInCollision->isSimulated()) continue;

                const double d = (output[i]->point[0]-origin)*direction;
                if (d<0.0 || d>maxLength) continue;
                if (result.body == NULL || d < result.rayLength)
                {
                    result.body=modelInCollision;
                    result.indexCollisionElement = output[i]->elem.first.getIndex();
                    result.point = output[i]->point[0];
                    result.dist  = (output[i]->point[1]-output[i]->point[0]).norm();
                    result.rayLength  = d;
                }
            }
        }
    }
    return result;
}

}
}

