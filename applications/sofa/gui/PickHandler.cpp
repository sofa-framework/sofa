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
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/helper/system/gl.h>
#include <sofa/simulation/common/Simulation.h>

#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/SphereModel.h>



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
        delete mouseNode;
        mouseNode = NULL;
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
    static bool firstTime=true;
    if (firstTime)
    {

        _fboParams.depthInternalformat = GL_DEPTH_COMPONENT24;
#ifdef GL_VERSION_3_0
        if (GLEW_VERSION_3_0)
        {
            _fboParams.colorInternalformat = GL_RGBA32F;
        }
        else
#endif
        {
            _fboParams.colorInternalformat = GL_RGBA16;
        }
        _fboParams.colorFormat         = GL_RGBA;
        _fboParams.colorType           = GL_FLOAT;

        _fbo.setFormat(_fboParams);
        firstTime=false;
    }
    _fbo.init(width,height);
    _fboAllocated = true;
}

void PickHandler::destroySelectionBuffer()
{
    /*called when shift key is released */
    assert(_fboAllocated);
    _fbo.destroy();
    _fboAllocated = false;
}


void PickHandler::init()
{
    if(mouseNode)
    {
        mouseNode->execute<sofa::simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        delete mouseNode;
        mouseNode = NULL;
    }
    std::vector< ComponentMouseInteraction *>::iterator it;
    for( it = instanceComponents.begin(); it != instanceComponents.end(); ++it)
    {
        if(*it != NULL ) delete *it;
    }
    instanceComponents.clear();

    //get a node of scene (root), create a new child (mouseNode), config it, then detach it from scene by default
    Node *root = dynamic_cast<Node*>(simulation::getSimulation()->getContext());
    mouseNode = root->createChild("Mouse");

    mouseContainer = new MouseContainer; mouseContainer->resize(1);
    mouseContainer->setName("MousePosition");
    mouseNode->addObject(mouseContainer);


    mouseCollision = new MouseCollisionModel;
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
    simulation::getSimulation()->getContext()->get(pipeline, core::objectmodel::BaseContext::SearchRoot);

    useCollisions = (pipeline != NULL);
}

void PickHandler::reset()
{
    deactivateRay();
    mouseButton = NONE;
    for (unsigned int i=0; i<instanceComponents.size(); ++i) instanceComponents[i]->reset();
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


void PickHandler::activateRay(int width, int height)
{
    if (!interactorInUse)
    {
        Node *root = static_cast<Node*>(simulation::getSimulation()->getContext());
        root->addChild(mouseNode);
        interaction->attach(mouseNode);
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
                interaction->attach(mouseNode);
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
                interaction->attach(mouseNode);
            }
        }
    }

}


void PickHandler::updateRay(const sofa::defaulttype::Vector3 &position,const sofa::defaulttype::Vector3 &orientation)
{
    if (!interactorInUse) return;

    mouseCollision->getRay(0).setOrigin( position+orientation*interaction->mouseInteractor->getDistanceFromMouse() );
    mouseCollision->getRay(0).setDirection( orientation );
    simulation::MechanicalPropagatePositionVisitor(sofa::core::MechanicalParams::defaultInstance() /* PARAMS FIRST */, 0, sofa::core::VecCoordId::position(), true).execute(mouseCollision->getContext());

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
    simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, origin, direction, maxLength, 0 );
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

//WARNING: do not use this method with Ogre
component::collision::BodyPicked PickHandler::findCollisionUsingColourCoding(const defaulttype::Vector3& origin,
        const defaulttype::Vector3& direction)
{
    assert(_fboAllocated);
    BodyPicked result;
    result.dist =  0;
    sofa::defaulttype::Vec4f color;
    int x = mousePosition.x;
    int y = mousePosition.screenHeight - mousePosition.y;
    TriangleModel* tmodel;
    SphereModel* smodel;
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
    return result;
}





}
}

