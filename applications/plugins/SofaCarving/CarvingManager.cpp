/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "CarvingManager.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/core/topology/TopologicalMapping.h>
#include <SofaUserInteraction/TopologicalChangeManager.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(CarvingManager)

int CarvingManagerClass = core::RegisterObject("Manager handling carving operations between a tool and an object.")
.add< CarvingManager >()
;


CarvingManager::CarvingManager()
: f_toolModelPath( initData(&f_toolModelPath, "toolModelPath", "Tool model path"))
, f_surfaceModelPath( initData(&f_surfaceModelPath, "surfaceModelPath", "TriangleSetModel or SphereModel path"))
, f_carvingDistance( initData(&f_carvingDistance, 0.0, "carvingDistance", "Collision distance at which cavring will start. Equal to contactDistance by default."))
, active( initData(&active, false, "active", "Activate this object.\nNote that this can be dynamically controlled by using a key") )
, keyEvent( initData(&keyEvent, '1', "key", "key to press to activate this object until the key is released") )
, keySwitchEvent( initData(&keySwitchEvent, '4', "keySwitch", "key to activate this object until the key is pressed again") )
, mouseEvent( initData(&mouseEvent, true, "mouseEvent", "Activate carving with middle mouse button") )
, omniEvent( initData(&omniEvent, true, "omniEvent", "Activate carving with omni button") )
, toolCollisionModel(NULL)
, intersectionMethod(NULL)
, detectionNP(NULL)
, m_carvingReady(false)
{
    this->f_listening.setValue(true);
}


CarvingManager::~CarvingManager()
{
}


void CarvingManager::init()
{
    // Search for collision model corresponding to the tool.
    if (f_toolModelPath.getValue().empty())
    {
        toolCollisionModel = getContext()->get<core::CollisionModel>(core::objectmodel::Tag("CarvingTool"), core::objectmodel::BaseContext::SearchDown);
        if (!toolCollisionModel)
            toolCollisionModel = getContext()->get<core::CollisionModel>(core::objectmodel::BaseContext::SearchDown);
    }
    else
        toolCollisionModel = getContext()->get<core::CollisionModel>(f_toolModelPath.getValue());

    // Search for the surface collision model.
    if (f_surfaceModelPath.getValue().empty())
    {
        // we look for a CollisionModel relying on a TetrahedronSetTopology.
        std::vector<core::CollisionModel*> models;
        getContext()->get<core::CollisionModel>(&models, core::objectmodel::Tag("CarvingSurface"), core::objectmodel::BaseContext::SearchRoot);
    
        // extend the research to model without the tag. 
        if (models.empty())
            getContext()->get<core::CollisionModel>(&models, core::objectmodel::BaseContext::SearchRoot);


        // If topological mapping, iterate into child Node to find mapped topology
	    sofa::core::topology::TopologicalMapping* topoMapping;
        for (unsigned int i=0;i<models.size();++i)
        {
            core::CollisionModel* m = models[i];
            m->getContext()->get(topoMapping);
            if (topoMapping == NULL) continue;
                        
            surfaceCollisionModels.push_back(m);
        }
    }
    else
    {
        surfaceCollisionModels.push_back(getContext()->get<core::CollisionModel>(f_surfaceModelPath.getValue()));
    }

    intersectionMethod = getContext()->get<core::collision::Intersection>();
    detectionNP = getContext()->get<core::collision::NarrowPhaseDetection>();

    if (!f_carvingDistance.isSet())
        f_carvingDistance.setValue(intersectionMethod->getContactDistance());

    m_carvingReady = true;

    if (toolCollisionModel == NULL) { msg_error() << "toolCollisionModel not found"; m_carvingReady = false; }
    if (surfaceCollisionModels.empty()) { msg_error() << "CarvingManager: surfaceCollisionModels not found"; m_carvingReady = false; }
    if (intersectionMethod == NULL) { msg_error() << "CarvingManager: intersectionMethod not found"; m_carvingReady = false; }
    if (detectionNP == NULL) { msg_error() << "CarvingManager: NarrowPhaseDetection not found"; m_carvingReady = false; }
    
    if (m_carvingReady)
        msg_info() << "CarvingManager: init OK.";
}


void CarvingManager::reset()
{

}


void CarvingManager::doCarve()
{
    if (m_carvingReady == false)
        return;

    const bool continuous = intersectionMethod->useContinuous();
    const double dt       = getContext()->getDt();
    const int depth = 6;

    if (continuous)
        toolCollisionModel->computeContinuousBoundingTree(dt, depth);
    else
        toolCollisionModel->computeBoundingTree(depth);

    sofa::helper::vector<std::pair<core::CollisionModel*, core::CollisionModel*> > vectCMPair;
    for (int i = 0; i < surfaceCollisionModels.size(); i++)
    {
        if (continuous)
            surfaceCollisionModels[i]->computeContinuousBoundingTree(dt, depth);
        else
            surfaceCollisionModels[i]->computeBoundingTree(depth);

        vectCMPair.push_back(std::make_pair(surfaceCollisionModels[i]->getFirst(), toolCollisionModel->getFirst()));
    }

    detectionNP->setInstance(this);
    detectionNP->setIntersectionMethod(intersectionMethod);
    detectionNP->beginNarrowPhase();
    detectionNP->addCollisionPairs(vectCMPair);
    detectionNP->endNarrowPhase();
    
    const core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs = detectionNP->getDetectionOutputs();

    const ContactVector* contacts = NULL;
    core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputs.begin(); 
    
    for (it = detectionOutputs.begin(); it != detectionOutputs.end(); ++it)
    {
        contacts = dynamic_cast<const ContactVector*>(it->second);
        unsigned int ncontacts = 0;
        if (contacts != NULL)
            ncontacts = contacts->size();

        if (ncontacts == 0)
            continue;

        int nbelems = 0;
        helper::vector<int> elemsToRemove;

        for (unsigned int j = 0; j < ncontacts; ++j)
        {
            const ContactVector::value_type& c = (*contacts)[j];
            
            if (c.value < f_carvingDistance.getValue())
            {
                int triangleIdx = (c.elem.first.getCollisionModel() == toolCollisionModel ? c.elem.second.getIndex() : c.elem.first.getIndex());
                elemsToRemove.push_back(triangleIdx);
            }
        }

        sofa::helper::AdvancedTimer::stepBegin("CarveElems");
        if (!elemsToRemove.empty())
        {
            static TopologicalChangeManager manager;
            if (it->first.first == toolCollisionModel)
                nbelems += manager.removeItemsFromCollisionModel(it->first.second, elemsToRemove);
            else
                nbelems += manager.removeItemsFromCollisionModel(it->first.first, elemsToRemove);
        }
    }
    

    detectionNP->setInstance(NULL);
}

void CarvingManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        dmsg_info() << "GET KEY "<<ev->getKey();
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
    else if (sofa::core::objectmodel::MouseEvent * ev = dynamic_cast<sofa::core::objectmodel::MouseEvent*>(event))
    {
        if ((ev->getState() == sofa::core::objectmodel::MouseEvent::MiddlePressed) && (mouseEvent.getValue()))
        {
            active.setValue(true);
        }
        else
        if ((ev->getState() == sofa::core::objectmodel::MouseEvent::MiddleReleased) && (mouseEvent.getValue()))
        {
            active.setValue(false);
        }
    }
    else if (sofa::core::objectmodel::HapticDeviceEvent * ev = dynamic_cast<sofa::core::objectmodel::HapticDeviceEvent *>(event))
    {
        if (ev->getButtonState()==1) active.setValue(true);
        else if (ev->getButtonState()==0) active.setValue(false);
    }

    else if (/* simulation::AnimateEndEvent* ev = */ dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
        if (active.getValue())
            doCarve();
    }
}

} // namespace collision

} // namespace component

} // namespace sofa
