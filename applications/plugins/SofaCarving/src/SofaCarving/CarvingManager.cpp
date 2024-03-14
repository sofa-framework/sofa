/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <SofaCarving/CarvingManager.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/ScriptEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>

#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/gui/component/performer/TopologicalChangeManager.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::collision
{

const int CarvingManagerClass = core::RegisterObject("Manager handling carving operations between a tool and an object.")
.add< CarvingManager >()
;


CarvingManager::CarvingManager()
    : l_toolModel(initLink("toolModel", "link to the carving collision model, if not set, manager will search for a collision model with tag: CarvingTool."))
    , l_detectionNP(initLink("narrowPhaseDetection", "link to the narrow Phase Detection component, if not set, manager will search for it in root Node."))
    , d_surfaceModelPath( initData(&d_surfaceModelPath, "surfaceModelPath", "TriangleSetModel or SphereCollisionModel<sofa::defaulttype::Vec3Types> path"))
    , d_carvingDistance( initData(&d_carvingDistance, 0.0, "carvingDistance", "Collision distance at which cavring will start. Equal to contactDistance by default."))
    , d_active( initData(&d_active, false, "active", "Activate this object.\nNote that this can be dynamically controlled by using a key") )
    , d_keyEvent( initData(&d_keyEvent, '1', "key", "key to press to activate this object until the key is released") )
    , d_keySwitchEvent( initData(&d_keySwitchEvent, '4', "keySwitch", "key to activate this object until the key is pressed again") )
    , d_mouseEvent( initData(&d_mouseEvent, true, "mouseEvent", "Activate carving with middle mouse button") )
    , d_omniEvent( initData(&d_omniEvent, true, "omniEvent", "Activate carving with omni button") )
    , d_activatorName(initData(&d_activatorName, "button1", "activatorName", "Name to active the script event parsing. Will look for 'pressed' or 'release' keyword. For example: 'button1_pressed'"))
{
    this->f_listening.setValue(true);
}



void CarvingManager::init()
{
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Loading);

    // Search for collision model corresponding to the tool.
    if (!l_toolModel.get())
    {
        auto toolCollisionModel = getContext()->get<core::CollisionModel>(core::objectmodel::Tag("CarvingTool"), core::objectmodel::BaseContext::SearchRoot);
        if (toolCollisionModel != nullptr)
            l_toolModel.set(toolCollisionModel);
    }

    // Search for the surface collision model.
    if (d_surfaceModelPath.getValue().empty())
    {
        // We look for a CollisionModel identified with the CarvingSurface Tag.
        getContext()->get<core::CollisionModel>(&m_surfaceCollisionModels, core::objectmodel::Tag("CarvingSurface"), core::objectmodel::BaseContext::SearchRoot);
    }
    else
    {
        m_surfaceCollisionModels.push_back(getContext()->get<core::CollisionModel>(d_surfaceModelPath.getValue()));
    }

    // If no NarrowPhaseDetection is set using the link try to find the component
    if (l_detectionNP.get() == nullptr)
    {
        l_detectionNP.set(getContext()->get<core::collision::NarrowPhaseDetection>());
    }
    

    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);

    if (l_toolModel.get() == nullptr) 
    {
        msg_error() << "Tool Collision Model not found. Set the link to toolModel or set tag 'CarvingTool' to the right collision model."; 
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }

    if (m_surfaceCollisionModels.empty()) 
    { 
        msg_error() << "m_surfaceCollisionModels not found. Set tag 'CarvingSurface' to the right collision models."; 
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }

    if (l_detectionNP.get() == nullptr)
    { 
        msg_error() << "NarrowPhaseDetection not found. Add a NarrowPhaseDetection method in your scene and link it using narrowPhaseDetection field."; 
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}


void CarvingManager::doCarve()
{
    if (d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    // get the collision output
    const core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs = l_detectionNP.get()->getDetectionOutputs();
    if (detectionOutputs.size() == 0)
        return;

    SCOPED_TIMER("CarvingElems");

    // loop on the contact to get the one between the CarvingSurface and the CarvingTool collision model
    const SReal& carvDist = d_carvingDistance.getValue();
    auto toolCollisionModel = l_toolModel.get();
    const ContactVector* contacts = NULL;
    for (core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputs.begin(); it != detectionOutputs.end(); ++it)
    {
        sofa::core::CollisionModel* collMod1 = it->first.first;
        sofa::core::CollisionModel* collMod2 = it->first.second;
        sofa::core::CollisionModel* targetModel = nullptr;

        if (collMod1 == toolCollisionModel && collMod2->hasTag(sofa::core::objectmodel::Tag("CarvingSurface"))) {
            targetModel = collMod2;
        }
        else if (collMod2 == toolCollisionModel && collMod1->hasTag(sofa::core::objectmodel::Tag("CarvingSurface"))) {
            targetModel = collMod1;
        }
        else {
            continue;
        }

        contacts = dynamic_cast<const ContactVector*>(it->second);
        if (contacts == nullptr || contacts->size() == 0) { 
            continue; 
        }

        size_t ncontacts = 0;
        if (contacts != NULL) {
            ncontacts = contacts->size();
        }

        if (ncontacts == 0) {
            continue;
        }

        int nbelems = 0;
        type::vector<Index> elemsToRemove;

        for (size_t j = 0; j < ncontacts; ++j)
        {
            const ContactVector::value_type& c = (*contacts)[j];

            if (c.value < carvDist)
            {
                auto elementIdx = (c.elem.first.getCollisionModel() == toolCollisionModel ? c.elem.second.getIndex() : c.elem.first.getIndex());
                elemsToRemove.push_back(elementIdx);
            }
        }

        if (!elemsToRemove.empty())
        {
            static sofa::gui::component::performer::TopologicalChangeManager manager;
            nbelems += manager.removeItemsFromCollisionModel(targetModel, elemsToRemove);
        }
    }
}

void CarvingManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        dmsg_info() << "GET KEY "<<ev->getKey();
        if (ev->getKey() == d_keyEvent.getValue())
        {
            d_active.setValue(true);
        }
        else if (ev->getKey() == d_keySwitchEvent.getValue())
        {
            d_active.setValue(!d_active.getValue());
        }
    }
    else if (sofa::core::objectmodel::KeyreleasedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeyreleasedEvent*>(event))
    {
        if (ev->getKey() == d_keyEvent.getValue())
        {
            d_active.setValue(false);
        }
    }
    else if (sofa::core::objectmodel::MouseEvent * ev = dynamic_cast<sofa::core::objectmodel::MouseEvent*>(event))
    {
        if ((ev->getState() == sofa::core::objectmodel::MouseEvent::MiddlePressed) && (d_mouseEvent.getValue()))
        {
            d_active.setValue(true);
        }
        else if ((ev->getState() == sofa::core::objectmodel::MouseEvent::MiddleReleased) && (d_mouseEvent.getValue()))
        {
            d_active.setValue(false);
        }
    }
    else if (sofa::core::objectmodel::HapticDeviceEvent * ev = dynamic_cast<sofa::core::objectmodel::HapticDeviceEvent *>(event))
    {
        if (ev->getButtonState() == 1) { d_active.setValue(true); }
        else if (ev->getButtonState() == 0) { d_active.setValue(false); }
    }
    else if (sofa::core::objectmodel::ScriptEvent *ev = dynamic_cast<sofa::core::objectmodel::ScriptEvent *>(event))
    {
        const std::string& eventS = ev->getEventName();
        if (eventS.find(d_activatorName.getValue()) != std::string::npos && eventS.find("pressed") != std::string::npos) {
            d_active.setValue(true);
        }

        if (eventS.find(d_activatorName.getValue()) != std::string::npos && eventS.find("released") != std::string::npos) {
            d_active.setValue(false);
        }
    }
    else if (sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        if (d_active.getValue()) {
            doCarve();
        }
    }


}

} // namespace sofa::component::collision
