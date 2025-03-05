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
#pragma once

#include <SofaCarving/config.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>

#include <fstream>

namespace sofa::component::collision
{

/**
* The CarvingManager class will perform topological resection on a triangle surface (could be on top of tetrahedron topology)
* The tool performing the carving need to be represented by a collision model @sa toolCollisionModel
* The surface to be carved are also mapped on collision models @sa surfaceCollisionModels
* Detecting the collision is done using the scene Intersection and NarrowPhaseDetection pipeline.
*/
class SOFA_SOFACARVING_API CarvingManager : public core::behavior::BaseController
{
public:
	SOFA_CLASS(CarvingManager,sofa::core::behavior::BaseController);
    
    using ContactVector = type::vector<core::collision::DetectionOutput>;
    
    /// Sofa API init method of the component
    void init() override;

    /// Method to handle various event like keyboard or omni.
    void handleEvent(sofa::core::objectmodel::Event* event) override;

    /// Impl method that will compute the intersection and check if some element have to be removed.
    virtual void doCarve();


protected:
    /// Default constructor
    CarvingManager();

    /// Default destructor
    ~CarvingManager() override {};


public:
    /// Tool model path
    // link to the forceFeedBack component, if not set will search through graph and take first one encountered
    SingleLink<CarvingManager, core::CollisionModel, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_toolModel;

    // link to the scene detection Method component (Narrow phase only)
    SingleLink<CarvingManager, core::collision::NarrowPhaseDetection, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_detectionNP;

    /// TriangleSetModel or SphereCollisionModel<sofa::defaulttype::Vec3Types> path
    Data < std::string > d_surfaceModelPath;

    /// Collision distance at which carving will start. Equal to contactDistance by default.
    Data < SReal > d_carvingDistance;
    
    ///< Activate this object. Note that this can be dynamically controlled by using a key
    Data < bool > d_active;
    ///< key to press to activate this object until the key is released
    Data < char > d_keyEvent;
    ///< key to activate this object until the key is pressed again
    Data < char > d_keySwitchEvent;
    ///< Activate carving with middle mouse button
    Data < bool > d_mouseEvent;
    ///< Activate carving with omni button
    Data < bool > d_omniEvent;
    ///< Activate carving with string Event, the activator name has to be inside the script event. Will look for 'pressed' or 'release' keyword. For example: 'button1_pressed'
    Data < std::string > d_activatorName;
    
protected:
    // Pointer to the target object collision model
    std::vector<core::CollisionModel*> m_surfaceCollisionModels;
   
};

} // namespace sofa::component::collision
