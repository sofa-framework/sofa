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
#ifndef SOFA_COMPONENT_COLLISION_GRASPINGMANAGER_H
#define SOFA_COMPONENT_COLLISION_GRASPINGMANAGER_H
#include "config.h"

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/collision/ContactManager.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/core/behavior/BaseController.h>
#include <set>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_USER_INTERACTION_API GraspingManager : public core::behavior::BaseController
{
public:
    SOFA_CLASS(GraspingManager,sofa::core::behavior::BaseController);

    typedef TriangleModel::DataTypes DataTypes;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Real Real;

    typedef core::CollisionModel ToolModel;
    typedef core::behavior::MechanicalState<defaulttype::Vec1Types> ToolDOFs;

    Data < bool > active; ///< Activate this object. Note that this can be dynamically controlled by using a key
    Data < char > keyEvent; ///< key to press to activate this object until the key is released
    Data < char > keySwitchEvent; ///< key to activate this object until the key is pressed again
    Data < double > openAngle; ///< angle values to set when tool is opened
    Data < double > closedAngle; ///< angle values to set when tool is closed

protected:
    std::set<ToolModel*> modelTools;
    ToolDOFs* mstateTool;
    core::collision::ContactManager* contactManager;
    bool wasActive;


    GraspingManager();

    virtual ~GraspingManager();
public:
    virtual void init() override;

    virtual void reset() override;

    virtual void handleEvent(sofa::core::objectmodel::Event* event) override;

    virtual void doGrasp();

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
