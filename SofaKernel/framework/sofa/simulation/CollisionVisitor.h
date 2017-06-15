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
#ifndef SOFA_SIMULATION_COLLISIONACTION_H
#define SOFA_SIMULATION_COLLISIONACTION_H

#include <sofa/core/ExecParams.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/collision/Pipeline.h>
#ifdef SOFA_SMP
#include <sofa/core/collision/ParallelPipeline.h>
#endif

namespace sofa
{

namespace simulation
{


/// Compute collision reset, detection and response in one step
class SOFA_SIMULATION_CORE_API CollisionVisitor : public Visitor
{
public:
    CollisionVisitor(const core::ExecParams* params) :Visitor(params) {}

    virtual void fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet);

    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "collision"; }
    virtual const char* getClassName() const { return "CollisionVisitor"; }
};

/// Remove collision response from last step
class SOFA_SIMULATION_CORE_API CollisionResetVisitor : public CollisionVisitor
{

public:
    CollisionResetVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);
    virtual const char* getClassName() const { return "CollisionResetVisitor"; }
};

/// Compute collision detection
class SOFA_SIMULATION_CORE_API CollisionDetectionVisitor : public CollisionVisitor
{
public:
    CollisionDetectionVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);
    virtual const char* getClassName() const { return "CollisionDetectionVisitor"; }
};

/// Compute collision response
class SOFA_SIMULATION_CORE_API CollisionResponseVisitor : public CollisionVisitor
{
public:
    CollisionResponseVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);
    virtual const char* getClassName() const { return "CollisionResponseVisitor"; }
};

#ifdef SOFA_SMP
/// Compute collision reset, detection and response in one step
class SOFA_SIMULATION_CORE_API ParallelCollisionVisitor : public CollisionVisitor
{
public:
    ParallelCollisionVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    virtual void processCollisionPipeline(simulation::Node* node, core::collision::ParallelPipeline* obj);
    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);

    virtual const char* getClassName() const { return "ParallelCollisionVisitor"; }
};
#endif

} // namespace simulation

} // namespace sofa

#endif
