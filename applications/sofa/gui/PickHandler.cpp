/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/gui/PickHandler.h>

#include <SofaUserInteraction/ComponentMouseInteraction.h>
#include <SofaUserInteraction/RayContact.h>

#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/helper/system/gl.h>
#include <sofa/simulation/Simulation.h>

#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/SphereModel.h>

#include <iostream>
#include <limits>


namespace sofa
{
using namespace component::collision;


namespace gui
{

PickHandler::PickHandler():
    interactorInUse(false),
    mouseStatus(DEACTIVATED),
    mouseButton(NONE),
    mouseNode(NULL),
    mouseContainer(NULL),
    mouseCollision(NULL),
#ifndef SOFA_NO_OPENGL
    _fbo(true,true,true,false,0),
#endif
    renderCallback(NULL),
    pickingMethod(RAY_CASTING),
    _fboAllocated(false)
{
    operations[LEFT] = operations[MIDDLE] = operations[RIGHT] = NULL;
}


PickHandler::~PickHandler()
{
    for (unsigned int i=0; i<operations.size(); ++i)
    {
        delete operations[i];
    }
    if(mouseNode)
    {
        mouseNode->execute<sofa::simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        //delete mouseNode;
        mouseNode.reset();
    }
    std::vector< ComponentMouseInteraction *>::iterator it;
    for( it = instanceComponents.begin(); it != instanceComponents.end(); ++it)
    {
        if(*it != NULL ) delete *it;
    }
    instanceComponents.clear();


}

void PickHandler::allocateSelectionBuffer(int width, int height)
{
    /*called when shift key is pressed */
    assert(_fboAllocated == false );
#ifndef SOFA_NO_OPENGL
    static bool firstTime=true;
    if (firstTime)
    {
#if defined (SOFA_HAVE_GLEW)
        _fboParams.depthInternalformat = GL_DEPTH_COMPONENT24;
#if defined(GL_VERSION_3_0)
        if (GLEW_VERSION_3_0)
        {
            _fboParams.colorInternalformat = GL_RGBA32F;
        }
        else
#endif //  (GL_VERSION_3_0)
        {
            _fboParams.colorInternalformat = GL_RGBA16;
        }
        _fboParams.colorFormat         = GL_RGBA;
        _fboParams.colorType           = GL_FLOAT;

        _fbo.setFormat(_fboParams);
#endif //  (SOFA_HAVE_GLEW)
        firstTime=false;
    }
#if defined (SOFA_HAVE_GLEW)
    _fbo.init(width,height);
#endif //  (SOFA_HAVE_GLEW)
#endif /* SOFA_NO_OPENGL */
    _fboAllocated = true;
}

void PickHandler::destroySelectionBuffer()
{
    /*called when shift key is released */
    assert(_fboAllocated);
#ifndef SOFA_NO_OPENGL
#ifdef SOFA_HAVE_GLEW
    _fbo.destroy();
#endif // SOFA_HAVE_GLEW
#endif // SOFA_NO_OPENGL
    _fboAllocated = false;
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
    mouseCollision->getRay(0).setL(1000000);
    mouseCollision->setName("MouseCollisionModel");
    mouseNode->addObject(mouseCollision);


    mouseNode->init(sofa::core::ExecParams::defaultInstance());
    mouseContainer->init();
    mouseCollision->init();

    typedef component::collision::ComponentMouseInteraction::ComponentMouseInteractionFactory MouseFactory;
    const MouseFactory *factory = MouseFactory::getInstance();
    for (MouseFactory::const_iterator it = factory->begin(); it != factory->end(); ++it)
    {
        instanceComponents.push_back(it->second->createInstance(NULL));
    }
    interaction = instanceComponents.back();

    mouseNode->detachFromGraph();


    core::collision::Pipeline *pipeline;
    root->getContext()->get(pipeline, core::objectmodel::BaseContext::SearchRoot);

    useCollisions = (pipeline != NULL);
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
        mouseNode->execute<sofa::simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        mouseNode.reset();
    }
    std::vector< ComponentMouseInteraction *>::iterator it;
    for( it = instanceComponents.begin(); it != instanceComponents.end(); ++it)
    {
        if(*it != NULL ) delete *it;
    }
    instanceComponents.clear();

}

Operation *PickHandler::changeOperation(sofa::component::configurationsetting::MouseButtonSetting* setting)
{
    if (operations[setting->button.getValue().getSelectedId()]) delete operations[setting->button.getValue().getSelectedId()];
    Operation *mouseOp=OperationFactory::Instanciate(setting->getOperationType());
    mouseOp->configure(this,setting);
    operations[setting->button.getValue().getSelectedId()]=mouseOp;
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


void PickHandler::activateRay(int width, int height, core::objectmodel::BaseNode* root)
{
    if (!interactorInUse)
    {
        root->addChild(mouseNode);
        interaction->attach(mouseNode.get());
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
        mouseNode->detachFromGraph();

        operations[LEFT]->endOperation();
        operations[MIDDLE]->endOperation();
        operations[RIGHT]->endOperation();

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
                interaction->attach(mouseNode.get());
            }
        }
    }

}


void PickHandler::updateRay(const sofa::defaulttype::Vector3 &position,const sofa::defaulttype::Vector3 &orientation)
{
    if (!interactorInUse) return;

    mouseCollision->getRay(0).setOrigin( position+orientation*interaction->mouseInteractor->getDistanceFromMouse() );
    mouseCollision->getRay(0).setDirection( orientation );
    simulation::MechanicalPropagatePositionVisitor(sofa::core::MechanicalParams::defaultInstance(), 0, sofa::core::VecCoordId::position(), true).execute(mouseCollision->getContext());
    simulation::MechanicalPropagatePositionVisitor(sofa::core::MechanicalParams::defaultInstance(), 0, sofa::core::VecCoordId::freePosition(), true).execute(mouseCollision->getContext());

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
    BodyPicked result;
    switch( pickingMethod)
    {
    case RAY_CASTING:
        if (useCollisions)
        {
            component::collision::BodyPicked picked=findCollisionUsingPipeline();
            if (picked.body) result = picked;
            else result = findCollisionUsingBruteForce();
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    result.baryCoords = output[i]->baryCoords[1];
#endif
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
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                    result.baryCoords = output[i]->baryCoords[0];
#endif
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

    return findCollisionUsingBruteForce(origin, direction, maxLength, mouseNode->getRoot());
}

component::collision::BodyPicked PickHandler::findCollisionUsingColourCoding()
{
    const defaulttype::Vector3& origin          = mouseCollision->getRay(0).origin();
    const defaulttype::Vector3& direction       = mouseCollision->getRay(0).direction();

    return findCollisionUsingColourCoding(origin, direction);

}

component::collision::BodyPicked PickHandler::findCollisionUsingBruteForce(const defaulttype::Vector3& origin,
        const defaulttype::Vector3& direction,
        double maxLength, core::objectmodel::BaseNode* rootNode)
{
    BodyPicked result;
    // Look for particles hit by this ray
//  msg_info()<<"PickHandler::findCollisionUsingBruteForce" << std::endl;
    simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, maxLength, 0 );
    //core::objectmodel::BaseNode* rootNode = mouseNode->getRoot(); //sofa::simulation::getSimulation()->getContext()->toBaseNode();

    if (rootNode) picker.execute(rootNode->getContext());
    else
        dmsg_error("PickHandler") << "Root node not found.";

    picker.getClosestParticle( result.mstate, result.indexCollisionElement, result.point, result.rayLength );

    return result;
}

//WARNING: do not use this method with Ogre
component::collision::BodyPicked PickHandler::findCollisionUsingColourCoding(const defaulttype::Vector3& origin,
        const defaulttype::Vector3& direction)
{
    assert(_fboAllocated);
    BodyPicked result;
#ifndef SOFA_NO_OPENGL
    result.dist =  0;
    sofa::defaulttype::Vec4f color;
    int x = mousePosition.x;
    int y = mousePosition.screenHeight - mousePosition.y;
    TriangleModel* tmodel;
    SphereModel* smodel;
#ifdef SOFA_HAVE_GLEW
    _fbo.start();
    if(renderCallback)
    {
        renderCallback->render(ColourPickingVisitor::ENCODE_COLLISIONELEMENT );
        glReadPixels(x,y,1,1,_fboParams.colorFormat,_fboParams.colorType,color.elems);
        decodeCollisionElement(color,result);
        renderCallback->render(ColourPickingVisitor::ENCODE_RELATIVEPOSITION );
        glReadPixels(x,y,1,1,_fboParams.colorFormat,_fboParams.colorType,color.elems);
        if( ( tmodel = dynamic_cast<TriangleModel*>(result.body) ) != NULL )
        {
            decodePosition(result,color,tmodel,result.indexCollisionElement);
        }
        if( ( smodel = dynamic_cast<SphereModel*>(result.body)) != NULL)
        {
            decodePosition(result, color,smodel,result.indexCollisionElement);
        }
        result.rayLength = (result.point-origin)*direction;
    }
    _fbo.stop();
#endif // SOFA_HAVE_GLEW
#endif // SOFA_NO_OPENGL
    return result;
}





}
}

