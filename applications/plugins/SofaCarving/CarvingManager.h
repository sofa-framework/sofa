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
#ifndef SOFA_COMPONENT_COLLISION_CARVINGMANAGER_H
#define SOFA_COMPONENT_COLLISION_CARVINGMANAGER_H

#include <SofaCarving/config.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_SOFACARVING_API CarvingManager : public core::behavior::BaseController
{
public:
	SOFA_CLASS(CarvingManager,sofa::core::behavior::BaseController);

	typedef defaulttype::Vec3Types DataTypes;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Real Real;
    
    typedef helper::vector<core::collision::DetectionOutput> ContactVector;
    
    /// Sofa API methods
    virtual void init() override;

    virtual void reset() override;

    virtual void handleEvent(sofa::core::objectmodel::Event* event) override;

    virtual void doCarve();


protected:
    CarvingManager();

    virtual ~CarvingManager();


public:
    /// Tool model path
    Data < std::string > f_modelTool; 
    /// TriangleSetModel or SphereModel path
    Data < std::string > f_modelSurface;

    /// Collision distance at which cavring will start. Equal to contactDistance by default.
    Data < Real > f_carvingDistance;
    
    ///< Activate this object. Note that this can be dynamically controlled by using a key
    Data < bool > active;
    ///< key to press to activate this object until the key is released
    Data < char > keyEvent;
    ///< key to activate this object until the key is pressed again
    Data < char > keySwitchEvent;
    ///< Activate carving with middle mouse button
    Data < bool > mouseEvent;
    ///< Activate carving with omni button
    Data < bool > omniEvent;
    
protected:
    /// Pointer to the tool collision model
    core::CollisionModel* modelTool;

    // Pointer to the target object collision model
    std::vector<core::CollisionModel*> modelSurface;

    // Pointer to the scene intersection Method component
    core::collision::Intersection* intersectionMethod;
    // Pointer to the scene detection Method component (Narrow phase only)
    core::collision::NarrowPhaseDetection* detectionNP;

    // Bool to store the information if component has well be init and can be used.
    bool m_carvingReady;
    
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
