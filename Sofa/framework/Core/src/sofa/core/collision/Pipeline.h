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

#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/helper/set.h>
#include <sofa/type/vector.h>


namespace sofa::core
{

class CollisionModel;

namespace collision
{

class BroadPhaseDetection;
class CollisionGroupManager;
class ContactManager;
class Intersection;
class NarrowPhaseDetection;


/**
 * @brief Pipeline component gather list of collision models and control the sequence of computations
*/

class SOFA_CORE_API Pipeline : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(Pipeline, sofa::core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Pipeline)

protected:

    //sofa::type::vector<DetectionOutput*> detectionOutputs;

    sofa::type::vector<Intersection*> intersectionMethods;
    sofa::type::vector<BroadPhaseDetection*> broadPhaseDetections;
    sofa::type::vector<NarrowPhaseDetection*> narrowPhaseDetections;
    sofa::type::vector<ContactManager*> contactManagers;
    sofa::type::vector<CollisionGroupManager*> groupManagers;

    Intersection* intersectionMethod;
    BroadPhaseDetection* broadPhaseDetection;
    NarrowPhaseDetection* narrowPhaseDetection;
    ContactManager* contactManager;
    CollisionGroupManager* groupManager;

public:
//    typedef NarrowPhaseDetection::DetectionOutputMap DetectionOutputMap;
protected:
    Pipeline();

    ~Pipeline() override;
	
private:
    Pipeline(const Pipeline& n) = delete;
    Pipeline& operator=(const Pipeline& n) = delete;



public:
    void reset() override = 0;

    /// Remove collision response from last step
    virtual void computeCollisionReset()=0;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void computeCollisionDetection()=0;
    /// Add collision response in the simulation graph
    virtual void computeCollisionResponse()=0;

    void computeCollisions()
    {
        computeCollisionReset();
        computeCollisionDetection();
        computeCollisionResponse();
    }

    //sofa::type::vector<DetectionOutput*>& getDetectionOutputs() { return detectionOutputs; }

    /// Broad phase collision detection method accessor.
    const BroadPhaseDetection *getBroadPhaseDetection() const;

    /// Narrow phase collision detection method accessor.
    const NarrowPhaseDetection *getNarrowPhaseDetection() const;

    /// get the set of response available with the current collision pipeline
    virtual std::set< std::string > getResponseList() const=0;
protected:
    /// Remove collision response from last step
    virtual void doCollisionReset() = 0;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void doCollisionDetection(const sofa::type::vector<core::CollisionModel*>& collisionModels) = 0;
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse() = 0;

public:

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;
};

} // namespace collision
} // namespace sofa::core

