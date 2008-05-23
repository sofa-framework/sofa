#include "GraspingManager.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(GraspingManager)

int GraspingManagerClass = core::RegisterObject("Manager handling Grasping operations between a SphereModel and a TriangleSetModel relying on a TetrahedronSetTopology")
        .add< GraspingManager >()
        ;


GraspingManager::GraspingManager()
    : active( initData(&active, false, "active", "Activate this object.\nNote that this can be dynamically controlled by using a key") )
    , keyEvent( initData(&keyEvent, '1', "key", "key to press to activate this object until the key is released") )
    , keySwitchEvent( initData(&keySwitchEvent, '4', "keySwitch", "key to activate this object until the key is pressed again") )
    , openAngle( initData(&openAngle, 1.0, "openAngle", "angle values to set when tool is opened"))
    , closedAngle( initData(&closedAngle, 0.0, "closedAngle", "angle values to set when tool is closed"))
    , mstateTool(NULL)
    , contactManager(NULL)
    , wasActive(false)
{
    this->f_listening.setValue(true);
}

GraspingManager::~GraspingManager()
{
}

void GraspingManager::init()
{
    std::vector<ToolModel*> models;
    this->getContext()->get<ToolModel>(&models, core::objectmodel::BaseContext::SearchDown);
    for (unsigned int i=0; i<models.size(); i++)
    {
        if (models[i]->getContactResponse() == std::string("stick"))
        {
            modelTools.insert(models[i]);
        }
    }
    for (std::set<ToolModel*>::iterator it=modelTools.begin(), itend=modelTools.end(); it != itend; ++it)
        (*it)->setActive(false);
    std::cout << "GraspingManager: "<<modelTools.size()<<"/"<<models.size()<<" collision models selected."<<std::endl;
    mstateTool = getContext()->get<ToolDOFs>(core::objectmodel::BaseContext::SearchDown);
    if (mstateTool) std::cout << "GraspingManager: tool DOFs found"<<std::endl;
    contactManager = getContext()->get<core::componentmodel::collision::ContactManager>();
    if (contactManager) std::cout << "GraspingManager: ContactManager found"<<std::endl;
    std::cout << "GraspingManager: init OK." << std::endl;
}

void GraspingManager::reset()
{
}

void GraspingManager::doGrasp()
{
    bool newActive = active.getValue();
    if (newActive && ! wasActive)
    {
        std::cout << "GraspingManager activated" << std::endl;
        // activate CMs for one iteration
        for (std::set<ToolModel*>::iterator it=modelTools.begin(), itend=modelTools.end(); it != itend; ++it)
            (*it)->setActive(true);
    }
    else
    {
        // deactivate CMs
        for (std::set<ToolModel*>::iterator it=modelTools.begin(), itend=modelTools.end(); it != itend; ++it)
            (*it)->setActive(false);
    }

    if (!newActive && wasActive)
    {
        std::cout << "GraspingManager released" << std::endl;
        // clear existing contacts
        if (contactManager)
        {
            const core::componentmodel::collision::ContactManager::ContactVector& cv = contactManager->getContacts();
            for (core::componentmodel::collision::ContactManager::ContactVector::const_iterator it = cv.begin(), itend = cv.end(); it != itend; ++it)
            {
                core::componentmodel::collision::Contact* c = *it;
                if (modelTools.count(c->getCollisionModels().first) || modelTools.count(c->getCollisionModels().second))
                    c->setKeepAlive(false);
            }
        }
    }

    if (mstateTool)
    {
        double value = (newActive ? closedAngle.getValue() : openAngle.getValue());
        std::cout << value << std::endl;
        ToolDOFs::VecCoord& x = *mstateTool->getX();
        if (x.size() >= 1)
            x[0] = value;
        if (x.size() >= 2)
            x[1] = -value;
    }

    wasActive = newActive;
}

void GraspingManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        if (ev->getKey() == keyEvent.getValue())
        {
            active.setValue(true);
        }
        else if (ev->getKey() == keySwitchEvent.getValue())
        {
            active.setValue(!active.getValue());
        }
    }
    else if (sofa::core::objectmodel::KeyreleasedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeyreleasedEvent*>(event))
    {
        if (ev->getKey() == keyEvent.getValue())
        {
            active.setValue(false);
        }
    }
    else if (/* simulation::AnimateEndEvent* ev = */ dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
//        if (active.getValue())
        doGrasp();
    }
}

} // namespace collision

} // namespace component

} // namespace sofa
