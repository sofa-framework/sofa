/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/common/PickHandler.h>

#include <sofa/gui/component/performer/ComponentMouseInteraction.h>
#include <sofa/component/collision/response/contact/RayContact.h>

#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/collision/Pipeline.h>

#include <sofa/component/setting/MouseButtonSetting.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalPickParticlesVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPickParticlesVisitor;

#include <sofa/gui/component/performer/ComponentMouseInteraction.h>

#include <iostream>
#include <limits>

using sofa::gui::component::performer::BodyPicked;

namespace sofa::gui::common
{

PickHandler::PickHandler(double defaultLength):
    interactorInUse(false),
    mouseStatus(DEACTIVATED),
    mouseButton(NONE),
    mouseNode(nullptr),
    mouseContainer(nullptr),
    mouseCollision(nullptr),
    renderCallback(nullptr),
    pickingMethod(RAY_CASTING),
    m_defaultLength(defaultLength)
{
    operations[LEFT] = operations[MIDDLE] = operations[RIGHT] = nullptr;
}


PickHandler::~PickHandler()
{
    for (unsigned int i=0; i<operations.size(); ++i)
    {
        if (operations[i])
        {
            delete operations[i];
            operations[i] = nullptr;
        }
    }
    if(mouseNode)
    {
        mouseNode->execute<sofa::simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
        //delete mouseNode;
        mouseNode.reset();
    }
    std::vector< ComponentMouseInteraction *>::iterator it;
    for( it = instanceComponents.begin(); it != instanceComponents.end(); ++it)
    {
        if(*it != nullptr ) delete *it;
    }
    instanceComponents.clear();


}

void PickHandler::allocateSelectionBuffer(int width, int height)
{
    SOFA_UNUSED(width);
    SOFA_UNUSED(height);
}

void PickHandler::destroySelectionBuffer()
{

}


void PickHandler::init(core::objectmodel::BaseNode* root)
{


    //get a node of scene (root), create a new child (mouseNode), config it, then detach it from scene by default
    mouseNode = down_cast<simulation::Node>(root)->createChild("Mouse");

    mouseContainer = sofa::core::objectmodel::New<MouseContainer>(); mouseContainer->resize(1);
    mouseContainer->setName("MousePosition");
    mouseNode->addObject(mouseContainer);


    mouseCollision = sofa::core::objectmodel::New<MouseCollisionModel>();
    mouseCollision->setNbRay(1);
    mouseCollision->getRay(0).setL(m_defaultLength);
    mouseCollision->setName("MouseCollisionModel");
    mouseNode->addObject(mouseCollision);


    mouseNode->init(sofa::core::execparams::defaultInstance());
    mouseContainer->init();
    mouseCollision->init();

    typedef sofa::gui::component::performer::ComponentMouseInteraction::ComponentMouseInteractionFactory MouseFactory;
    const MouseFactory *factory = MouseFactory::getInstance();
    for (MouseFactory::const_iterator it = factory->begin(); it != factory->end(); ++it)
    {
        instanceComponents.push_back(it->second->createInstance(nullptr));
    }
    interaction = instanceComponents.back();

    mouseNode->detachFromGraph();


    core::collision::Pipeline *pipeline;
    root->getContext()->get(pipeline, core::objectmodel::BaseContext::SearchRoot);

    useCollisions = (pipeline != nullptr);
}

void PickHandler::reset()
{
    deactivateRay();
    mouseButton = NONE;
    for (unsigned int i=0; i<instanceComponents.size(); ++i) instanceComponents[i]->reset();
}

void PickHandler::unload()
{
    if(mouseNode)
    {
        mouseNode->execute<sofa::simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
        mouseNode.reset();
    }
    std::vector< ComponentMouseInteraction *>::iterator it;
    for( it = instanceComponents.begin(); it != instanceComponents.end(); ++it)
    {
        if(*it != nullptr ) delete *it;
    }
    instanceComponents.clear();

}

Operation *PickHandler::changeOperation(sofa::component::setting::MouseButtonSetting* setting)
{
    if (operations[setting->button.getValue().getSelectedId()])
    {
        delete operations[setting->button.getValue().getSelectedId()];
        operations[setting->button.getValue().getSelectedId()] = nullptr;
    }
    Operation *mouseOp=OperationFactory::Instanciate(setting->getOperationType());
    if (mouseOp)
    {
        mouseOp->configure(this,setting);
        operations[setting->button.getValue().getSelectedId()]=mouseOp;
    }

    return mouseOp;
}

Operation *PickHandler::changeOperation(MOUSE_BUTTON button, const std::string &op)
{
    if (operations[button])
    {
        delete operations[button];
        operations[button] = nullptr;
    }
    Operation *mouseOp=OperationFactory::Instanciate(op);
    mouseOp->configure(this,button);
    operations[button]=mouseOp;
    return mouseOp;
}


void PickHandler::activateRay(int width, int height, core::objectmodel::BaseNode* root)
{
    if (!interactorInUse)
    {
        if (mouseNode)
        {
            root->addChild(mouseNode);
            interaction->attach(mouseNode.get());
        }

        if( pickingMethod == SELECTION_BUFFER)
        {
            allocateSelectionBuffer(width,height);
        }
        interactorInUse=true;
    }
}

void PickHandler::deactivateRay()
{
    if (interactorInUse )
    {
        if (mouseNode)
            mouseNode->detachFromGraph();

        if (operations[LEFT]) operations[LEFT]->endOperation();            
        if (operations[MIDDLE]) operations[MIDDLE]->endOperation();
        if (operations[RIGHT]) operations[RIGHT]->endOperation();

        interaction->detach();
        if( pickingMethod == SELECTION_BUFFER)
        {
            destroySelectionBuffer();
        }
        interactorInUse=false;
    }

}




bool PickHandler::needToCastRay()
{
    return !getInteraction()->mouseInteractor->isMouseAttached();
}


void PickHandler::setCompatibleInteractor()
{
    if (!lastPicked.body && !lastPicked.mstate) return;

    if (lastPicked.body)
    {
        if (interaction->isCompatible(lastPicked.body->getContext())) return;
        for (unsigned int i=0; i<instanceComponents.size(); ++i)
        {
            if (instanceComponents[i] != interaction &&
                instanceComponents[i]->isCompatible(lastPicked.body->getContext()))
            {
                interaction->detach();
                interaction = instanceComponents[i];
                if (mouseNode) 
                    interaction->attach(mouseNode.get());
            }
        }
    }
    else
    {
        if (interaction->isCompatible(lastPicked.mstate->getContext())) return;
        for (unsigned int i=0; i<instanceComponents.size(); ++i)
        {
            if (instanceComponents[i] != interaction &&
                instanceComponents[i]->isCompatible(lastPicked.mstate->getContext()))
            {
                interaction->detach();
                interaction = instanceComponents[i];
                if (mouseNode) 
                    interaction->attach(mouseNode.get());
            }
        }
    }

}


void PickHandler::updateRay(const sofa::type::Vec3 &position,const sofa::type::Vec3 &orientation)
{
    if (!interactorInUse || !mouseCollision) return;

    mouseCollision->getRay(0).setOrigin( position+orientation*interaction->mouseInteractor->getDistanceFromMouse() );
    mouseCollision->getRay(0).setDirection( orientation );
    MechanicalPropagateOnlyPositionVisitor(sofa::core::mechanicalparams::defaultInstance(), 0, sofa::core::VecCoordId::position()).execute(mouseCollision->getContext());
    MechanicalPropagateOnlyPositionVisitor(sofa::core::mechanicalparams::defaultInstance(), 0, sofa::core::VecCoordId::freePosition()).execute(mouseCollision->getContext());

    if (needToCastRay())
    {
        lastPicked=findCollision();
        setCompatibleInteractor();
        interaction->mouseInteractor->setMouseRayModel(mouseCollision.get());
        interaction->mouseInteractor->setBodyPicked(lastPicked);
        for (unsigned int i=0; i<callbacks.size(); ++i)
        {
            callbacks[i]->execute(lastPicked);
        }
    }
    

    if(mouseButton != NONE)
    {
        switch (mouseStatus)
        {
        case PRESSED:
        {
            if (operations[mouseButton]) operations[mouseButton]->start();
            mouseStatus=ACTIVATED;
            break;
        }
        case RELEASED:
        {
            if (operations[mouseButton]) operations[mouseButton]->end();
            mouseStatus=DEACTIVATED;
            break;
        }
        case ACTIVATED:
        {
            if (operations[mouseButton]) operations[mouseButton]->execution();
        }
        case DEACTIVATED:
        {
        }
        }
    }
    for (unsigned int i=0; i<operations.size(); ++i)
    {
        if (operations[i])
            operations[i]->wait();
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


BodyPicked PickHandler::findCollision()
{
    BodyPicked result;
    switch( pickingMethod)
    {
    case RAY_CASTING:
        if (useCollisions)
        {
            const BodyPicked picked = findCollisionUsingPipeline();
            if (picked.body) 
                result = picked;
            else 
                result = findCollisionUsingBruteForce();
        }
        else
            result = findCollisionUsingBruteForce();
        break;
    case SELECTION_BUFFER:
        result = findCollisionUsingColourCoding();
        break;
    default:
        assert(false);
    }
    return result;
}

BodyPicked PickHandler::findCollisionUsingPipeline()
{
    BodyPicked result;

    if (!mouseCollision) {
        dmsg_error("PickHandler") << "No mouseCollision.";
        return result;
    }

    const type::Vec3& origin          = mouseCollision->getRay(0).origin();
    const type::Vec3& direction       = mouseCollision->getRay(0).direction();
    const double& maxLength              = mouseCollision->getRay(0).l();
    
    const auto &contacts = mouseCollision->getContacts();
    for (auto it=contacts.cbegin(); it != contacts.cend(); ++it)
    {

        const sofa::type::vector<core::collision::DetectionOutput*>& output = (*it)->getDetectionOutputs();
        sofa::core::CollisionModel *modelInCollision;
        for (unsigned int i=0; i<output.size(); ++i)
        {

            if (output[i]->elem.first.getCollisionModel() == mouseCollision)
            {
                modelInCollision = output[i]->elem.second.getCollisionModel();
                if (!modelInCollision->isSimulated()) continue;


                const double d = (output[i]->point[1]-origin)*direction;
                if (d<0.0 || d>maxLength) continue;
                if (result.body == nullptr || d < result.rayLength)
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
                if (result.body == nullptr || d < result.rayLength)
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

BodyPicked PickHandler::findCollisionUsingBruteForce()
{
    const type::Vec3& origin          = mouseCollision->getRay(0).origin();
    const type::Vec3& direction       = mouseCollision->getRay(0).direction();
    const double& maxLength                     = mouseCollision->getRay(0).l();

    return findCollisionUsingBruteForce(origin, direction, maxLength, mouseNode->getRoot());
}

BodyPicked PickHandler::findCollisionUsingColourCoding()
{
    const type::Vec3& origin          = mouseCollision->getRay(0).origin();
    const type::Vec3& direction       = mouseCollision->getRay(0).direction();

    return findCollisionUsingColourCoding(origin, direction);

}

BodyPicked PickHandler::findCollisionUsingBruteForce(const type::Vec3& origin,
        const type::Vec3& direction,
        double maxLength, core::objectmodel::BaseNode* rootNode)
{
    BodyPicked result;
    // Look for particles hit by this ray
//  msg_info()<<"PickHandler::findCollisionUsingBruteForce" << std::endl;
    MechanicalPickParticlesVisitor picker(sofa::core::execparams::defaultInstance(), origin, direction, maxLength, 0 );
    //core::objectmodel::BaseNode* rootNode = mouseNode->getRoot(); //sofa::simulation::getSimulation()->getContext()->toBaseNode();

    if (rootNode) picker.execute(rootNode->getContext());
    else
        dmsg_error("PickHandler") << "Root node not found.";

    picker.getClosestParticle( result.mstate, result.indexCollisionElement, result.point, result.rayLength );

    return result;
}

//WARNING: do not use this method with Ogre
BodyPicked PickHandler::findCollisionUsingColourCoding(const type::Vec3& origin,
        const type::Vec3& direction)
{
    SOFA_UNUSED(origin);
    SOFA_UNUSED(direction);

    const BodyPicked result;

    msg_error("PickHandler") << "findCollisionUsingColourCoding not implemented!";

    return result;
}

} // namespace sofa::gui::common
