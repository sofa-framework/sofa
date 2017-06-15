/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
    
	typedef core::CollisionModel ToolModel;
    typedef helper::vector<core::collision::DetectionOutput> ContactVector;

    Data < std::string > f_modelTool;
    Data < std::string > f_modelSurface;
    Data < Real > f_minDistance;
    Data < Real > f_maxDistance;
    Data < Real > f_edgeDistance;
    
    
    Data < bool > active;
    Data < char > keyEvent;
    Data < char > keySwitchEvent;
    Data < bool > mouseEvent;
    Data < bool > omniEvent;
    
protected:
    ToolModel* modelTool;
    core::CollisionModel* modelSurface;
    core::collision::Intersection* intersectionMethod;
    core::collision::NarrowPhaseDetection* detectionNP;


    CarvingManager();

    virtual ~CarvingManager();
public:
    virtual void init();

    virtual void reset();
    
    virtual void handleEvent(sofa::core::objectmodel::Event* event);

    virtual void doCarve();

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
