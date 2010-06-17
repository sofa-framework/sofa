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

#include <sofa/component/collision/ComponentMouseInteraction.h>
#include <sofa/component/collision/RayContact.h>

#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#include <iostream>





namespace sofa
{
using namespace component::collision;


namespace gui
{

PickHandler::PickHandler():interactorInUse(false), mouseStatus(DEACTIVATED),mouseButton(NONE),renderCallback(NULL)
{
    operations[LEFT] = operations[MIDDLE] = operations[RIGHT] = NULL;

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
    for (unsigned int i=0; i<operations.size(); ++i)
    {
        delete operations[i];
    }

    if( renderCallback ) delete renderCallback;
//       for (unsigned int i=0;i<instanceComponents.size();++i) delete instanceComponents[i];
}

void PickHandler::init()
{
    core::collision::Pipeline *pipeline;
    simulation::getSimulation()->getContext()->get(pipeline, core::objectmodel::BaseContext::SearchRoot);

    useCollisions = (pipeline != NULL);

    _fboParams.depthInternalformat = GL_DEPTH_COMPONENT24;
    _fboParams.colorInternalformat = GL_RGBA32F;
    _fboParams.colorFormat         = GL_RGBA;
    _fboParams.colorType           = GL_FLOAT;

    _fbo.setFormat(_fboParams);
    _fbo.init(GL_MAX_TEXTURE_SIZE,GL_MAX_TEXTURE_SIZE);

}

void PickHandler::reset()
{
    activateRay(false);
    mouseButton = NONE;
    for (unsigned int i=0; i<instanceComponents.size(); ++i) instanceComponents[i]->reset();
}

Operation *PickHandler::changeOperation(sofa::component::configurationsetting::MouseButtonSetting* setting)
{
    if (operations[setting->getButton()]) delete operations[setting->getButton()];
    Operation *mouseOp=OperationFactory::Instanciate(setting->getOperationType());
    mouseOp->configure(this,setting);
    operations[setting->getButton()]=mouseOp;
    return mouseOp;
}

Operation *PickHandler::changeOperation(MOUSE_BUTTON button, const std::string &op)
{
    if (operations[button]) delete operations[button];
    Operation *mouseOp=OperationFactory::Instanciate(op);
    mouseOp->configure(this,button);
    operations[button]=mouseOp;
    return mouseOp;
}


void PickHandler::activateRay(bool act)
{
    if (interactorInUse && !act)
    {
        mouseNode->detachFromGraph();


        operations[LEFT]->endOperation();
        operations[MIDDLE]->endOperation();
        operations[RIGHT]->endOperation();

        interaction->deactivate();

        interactorInUse=false;
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
    if (!lastPicked.body && !lastPicked.mstate) return;

    if (lastPicked.body)
    {
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
    else
    {
        if (interaction->isCompatible(lastPicked.mstate->getContext())) return;
        for (unsigned int i=0; i<instanceComponents.size(); ++i)
        {
            if (instanceComponents[i] != interaction &&
                instanceComponents[i]->isCompatible(lastPicked.mstate->getContext()))
            {
                interaction->deactivate();
                interaction = instanceComponents[i];
                interaction->activate();
            }
        }
    }

}


void PickHandler::updateRay(const sofa::defaulttype::Vector3 &position,const sofa::defaulttype::Vector3 &orientation)
{
    if (!interactorInUse) return;

    mouseCollision->getRay(0).origin() = position+orientation*interaction->mouseInteractor->getDistanceFromMouse();
    mouseCollision->getRay(0).direction() = orientation;
    simulation::MechanicalPropagatePositionVisitor().execute(mouseCollision->getContext());

    if (needToCastRay())
    {
        lastPicked=findCollision();
        setCompatibleInteractor();
        interaction->mouseInteractor->setMouseRayModel(mouseCollision);
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
        }
        }
    }

    for (unsigned int i=0; i<operations.size(); ++i)
    {
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


component::collision::BodyPicked PickHandler::findCollision()
{
    if (useCollisions)
    {
        component::collision::BodyPicked picked=findCollisionUsingPipeline();
        if (picked.body) return picked;
    }
    return findCollisionUsingBruteForce();
    //return findCollisionUsingColourCoding();
}

component::collision::BodyPicked PickHandler::findCollisionUsingPipeline()
{
    const defaulttype::Vector3& origin          = mouseCollision->getRay(0).origin();
    const defaulttype::Vector3& direction       = mouseCollision->getRay(0).direction();
    const double& maxLength                     = mouseCollision->getRay(0).l();

    BodyPicked result;
    const std::set< sofa::component::collision::BaseRayContact*> &contacts = mouseCollision->getContacts();
    for (std::set< sofa::component::collision::BaseRayContact*>::const_iterator it=contacts.begin(); it != contacts.end(); ++it)
    {

        const sofa::helper::vector<core::collision::DetectionOutput*>& output = (*it)->getDetectionOutputs();
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

component::collision::BodyPicked PickHandler::findCollisionUsingBruteForce()
{
    const defaulttype::Vector3& origin          = mouseCollision->getRay(0).origin();
    const defaulttype::Vector3& direction       = mouseCollision->getRay(0).direction();
    const double& maxLength                     = mouseCollision->getRay(0).l();

    return findCollisionUsingBruteForce(origin, direction, maxLength);
}

component::collision::BodyPicked PickHandler::findCollisionUsingColourCoding()
{
    const defaulttype::Vector3& origin          = mouseCollision->getRay(0).origin();
    const defaulttype::Vector3& direction       = mouseCollision->getRay(0).direction();

    return findCollisionUsingColourCoding(origin, direction);

}


component::collision::BodyPicked PickHandler::findCollisionUsingBruteForce(const defaulttype::Vector3& origin,
        const defaulttype::Vector3& direction,
        double maxLength)
{
    BodyPicked result;
    // Look for particles hit by this ray
    simulation::MechanicalPickParticlesVisitor picker(origin, direction, maxLength, 0);
    core::objectmodel::BaseNode* rootNode = dynamic_cast<core::objectmodel::BaseNode*>(sofa::simulation::getSimulation()->getContext());

    if (rootNode) picker.execute(rootNode->getContext());
    else std::cerr << "ERROR: root node not found." << std::endl;

    if (!picker.particles.empty())
    {
        core::behavior::BaseMechanicalState *mstate = picker.particles.begin()->second.first;
        result.mstate=mstate;
        result.indexCollisionElement = picker.particles.begin()->second.second;
        result.point[0] = mstate->getPX(result.indexCollisionElement);
        result.point[1] = mstate->getPY(result.indexCollisionElement);
        result.point[2] = mstate->getPZ(result.indexCollisionElement);
        result.dist =  0;
        result.rayLength = (result.point-origin)*direction;
    }
    return result;
}

component::collision::BodyPicked PickHandler::findCollisionUsingColourCoding(const defaulttype::Vector3& origin,
        const defaulttype::Vector3& direction)
{
    BodyPicked result;
    static const float threshold = 0.001f;

    sofa::defaulttype::Vec4f color;
    int x = mousePosition.screenWidth -  mousePosition.x;
    int y = mousePosition.screenHeight - mousePosition.y;
    _fbo.start();
    if(renderCallback)
    {
        renderCallback->render();
    }

    glReadPixels(x,y,1,1,_fboParams.colorFormat,_fboParams.colorType,color.elems);

    _fbo.stop();

    result = _decodeColour(color, origin, direction);

    return result;

}

component::collision::BodyPicked PickHandler::_decodeColour(const sofa::defaulttype::Vec4f& colour,
        const defaulttype::Vector3& origin,
        const defaulttype::Vector3& direction)
{
    using namespace core::objectmodel;
    using namespace core::behavior;
    using namespace sofa::defaulttype;
    static const float threshold = 0.001f;

    component::collision::BodyPicked result;

    result.dist =  0;

    if( colour[0] > threshold )
    {

        helper::vector<core::CollisionModel*> listCollisionModel;
        sofa::simulation::getSimulation()->getContext()->get<core::CollisionModel>(&listCollisionModel,BaseContext::SearchRoot);
        const int totalCollisionModel = listCollisionModel.size();

        const int indexListCollisionModel = (int) ( colour[0] * (float)totalCollisionModel + 0.5) - 1;
        result.body = listCollisionModel[indexListCollisionModel];
        result.indexCollisionElement = (int) ( colour[1] * result.body->getSize() );

        if( colour[2] < threshold && colour[3] < threshold )
        {
            /* no barycentric weights */
            core::behavior::BaseMechanicalState *mstate;
            result.body->getContext()->get(mstate,BaseContext::Local);
            if(mstate)
            {
                result.point[0] = mstate->getPX(result.indexCollisionElement);
                result.point[1] = mstate->getPY(result.indexCollisionElement);
                result.point[2] = mstate->getPZ(result.indexCollisionElement);
            }
        }

        result.rayLength = (result.point-origin)*direction;
    }

    return result;
}




}
}

