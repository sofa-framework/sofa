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

	typedef defaulttype::Vec3Types DataTypes;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Real Real;
    
    typedef helper::vector<core::collision::DetectionOutput> ContactVector;
    
    /// Sofa API init method of the component
    virtual void init() override;
    /// Sofa API reset method of the component
    virtual void reset() override;

    /// Method to handle various event like keyboard or omni.
    virtual void handleEvent(sofa::core::objectmodel::Event* event) override;

    /// Impl method that will compute the intersection and check if some element have to be removed.
    virtual void doCarve();


protected:
    /// Default constructor
    CarvingManager();

    /// Default destructor
    virtual ~CarvingManager();


public:
    /// Tool model path
    Data < std::string > d_toolModelPath; 
    /// TriangleSetModel or SphereModel path
    Data < std::string > d_surfaceModelPath;

    /// Collision distance at which cavring will start. Equal to contactDistance by default.
    Data < Real > d_carvingDistance;
    
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
    
protected:
    /// Pointer to the tool collision model
    core::CollisionModel* m_toolCollisionModel;

    // Pointer to the target object collision model
    std::vector<core::CollisionModel*> m_surfaceCollisionModels;

    // Pointer to the scene intersection Method component
    core::collision::Intersection* m_intersectionMethod;
    // Pointer to the scene detection Method component (Narrow phase only)
    core::collision::NarrowPhaseDetection* m_detectionNP;

    // Bool to store the information if component has well be init and can be used.
    bool m_carvingReady;
    
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
