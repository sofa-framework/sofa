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
    : d_toolModelPath( initData(&d_toolModelPath, "toolModelPath", "Tool model path"))
    , d_surfaceModelPath( initData(&d_surfaceModelPath, "surfaceModelPath", "TriangleSetModel or SphereModel path"))
    , d_carvingDistance( initData(&d_carvingDistance, 0.0, "carvingDistance", "Collision distance at which cavring will start. Equal to contactDistance by default."))
    , d_active( initData(&d_active, false, "active", "Activate this object.\nNote that this can be dynamically controlled by using a key") )
    , d_keyEvent( initData(&d_keyEvent, '1', "key", "key to press to activate this object until the key is released") )
    , d_keySwitchEvent( initData(&d_keySwitchEvent, '4', "keySwitch", "key to activate this object until the key is pressed again") )
    , d_mouseEvent( initData(&d_mouseEvent, true, "mouseEvent", "Activate carving with middle mouse button") )
    , d_omniEvent( initData(&d_omniEvent, true, "omniEvent", "Activate carving with omni button") )
    , m_toolCollisionModel(NULL)
    , m_intersectionMethod(NULL)
    , m_detectionNP(NULL)
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
    if (d_toolModelPath.getValue().empty())
    {
        m_toolCollisionModel = getContext()->get<core::CollisionModel>(core::objectmodel::Tag("CarvingTool"), core::objectmodel::BaseContext::SearchDown);
        if (!m_toolCollisionModel)
            m_toolCollisionModel = getContext()->get<core::CollisionModel>(core::objectmodel::BaseContext::SearchDown);
    }
    else
        m_toolCollisionModel = getContext()->get<core::CollisionModel>(d_toolModelPath.getValue());

    // Search for the surface collision model.
    if (d_surfaceModelPath.getValue().empty())
    {
        // we look for a CollisionModel relying on a TetrahedronSetTopology.
        std::vector<core::CollisionModel*> models;
        getContext()->get<core::CollisionModel>(&models, core::objectmodel::Tag("CarvingSurface"), core::objectmodel::BaseContext::SearchRoot);
    
        // extend the research to model without the tag. 
        if (models.empty())
            getContext()->get<core::CollisionModel>(&models, core::objectmodel::BaseContext::SearchRoot);


        // If topological mapping, iterate into child Node to find mapped topology
	    sofa::core::topology::TopologicalMapping* topoMapping;
        for (size_t i=0;i<models.size();++i)
        {
            core::CollisionModel* m = models[i];
            m->getContext()->get(topoMapping);
            if (topoMapping == NULL) continue;
                        
            m_surfaceCollisionModels.push_back(m);
        }
    }
    else
    {
        m_surfaceCollisionModels.push_back(getContext()->get<core::CollisionModel>(d_surfaceModelPath.getValue()));
    }

    m_intersectionMethod = getContext()->get<core::collision::Intersection>();
    m_detectionNP = getContext()->get<core::collision::NarrowPhaseDetection>();

    if (!d_carvingDistance.isSet())
        d_carvingDistance.setValue(m_intersectionMethod->getContactDistance());

    m_carvingReady = true;

    if (m_toolCollisionModel == NULL) { msg_error() << "m_toolCollisionModel not found"; m_carvingReady = false; }
    if (m_surfaceCollisionModels.empty()) { msg_error() << "CarvingManager: m_surfaceCollisionModels not found"; m_carvingReady = false; }
    if (m_intersectionMethod == NULL) { msg_error() << "CarvingManager: m_intersectionMethod not found"; m_carvingReady = false; }
    if (m_detectionNP == NULL) { msg_error() << "CarvingManager: NarrowPhaseDetection not found"; m_carvingReady = false; }
    
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

    const bool continuous = m_intersectionMethod->useContinuous();
    const double dt       = getContext()->getDt();
    const int depth = 6;

    if (continuous)
        m_toolCollisionModel->computeContinuousBoundingTree(dt, depth);
    else
        m_toolCollisionModel->computeBoundingTree(depth);

    sofa::helper::vector<std::pair<core::CollisionModel*, core::CollisionModel*> > vectCMPair;
    for (size_t i = 0; i < m_surfaceCollisionModels.size(); i++)
    {
        if (continuous)
            m_surfaceCollisionModels[i]->computeContinuousBoundingTree(dt, depth);
        else
            m_surfaceCollisionModels[i]->computeBoundingTree(depth);

        vectCMPair.push_back(std::make_pair(m_surfaceCollisionModels[i]->getFirst(), m_toolCollisionModel->getFirst()));
    }

    m_detectionNP->setInstance(this);
    m_detectionNP->setIntersectionMethod(m_intersectionMethod);
    m_detectionNP->beginNarrowPhase();
    m_detectionNP->addCollisionPairs(vectCMPair);
    m_detectionNP->endNarrowPhase();
    
    const core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs = m_detectionNP->getDetectionOutputs();

    const ContactVector* contacts = NULL;
    core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputs.begin(); 
    
    for (it = detectionOutputs.begin(); it != detectionOutputs.end(); ++it)
    {
        contacts = dynamic_cast<const ContactVector*>(it->second);
        size_t ncontacts = 0;
        if (contacts != NULL)
            ncontacts = contacts->size();

        if (ncontacts == 0)
            continue;

        int nbelems = 0;
        helper::vector<int> elemsToRemove;

        for (size_t j = 0; j < ncontacts; ++j)
        {
            const ContactVector::value_type& c = (*contacts)[j];
            
            if (c.value < d_carvingDistance.getValue())
            {
                int triangleIdx = (c.elem.first.getCollisionModel() == m_toolCollisionModel ? c.elem.second.getIndex() : c.elem.first.getIndex());
                elemsToRemove.push_back(triangleIdx);
            }
        }

        sofa::helper::AdvancedTimer::stepBegin("CarveElems");
        if (!elemsToRemove.empty())
        {
            static TopologicalChangeManager manager;
            if (it->first.first == m_toolCollisionModel)
                nbelems += manager.removeItemsFromCollisionModel(it->first.second, elemsToRemove);
            else
                nbelems += manager.removeItemsFromCollisionModel(it->first.first, elemsToRemove);
        }
    }
    

    m_detectionNP->setInstance(NULL);
}

void CarvingManager::handleEvent(sofa::core::objectmodel::Event* event)
{
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
        else
        if ((ev->getState() == sofa::core::objectmodel::MouseEvent::MiddleReleased) && (d_mouseEvent.getValue()))
        {
            d_active.setValue(false);
        }
    }
    else if (sofa::core::objectmodel::HapticDeviceEvent * ev = dynamic_cast<sofa::core::objectmodel::HapticDeviceEvent *>(event))
    {
        if (ev->getButtonState()==1) d_active.setValue(true);
        else if (ev->getButtonState()==0) d_active.setValue(false);
    }

    else if (/* simulation::AnimateEndEvent* ev = */ dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
        if (d_active.getValue())
            doCarve();
    }
}

} // namespace collision

} // namespace component

} // namespace sofa
