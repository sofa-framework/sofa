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
#ifndef SOFA_SIMULATION_COLLISIONACTION_H
#define SOFA_SIMULATION_COLLISIONACTION_H


#include <sofa/simulation/Visitor.h>


namespace sofa::simulation
{


/// Compute collision reset, detection and response in one step
class SOFA_SIMULATION_CORE_API CollisionVisitor : public Visitor
{
public:
    CollisionVisitor(const core::ExecParams* params) :Visitor(params) , m_primitiveTestCount(0) {}

    virtual void fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet);

    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);

    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "collision"; }
    const char* getClassName() const override { return "CollisionVisitor"; }

    size_t getPrimitiveTestCount() const {return m_primitiveTestCount;}
private:
    size_t m_primitiveTestCount;
};

/// Remove collision response from last step
class SOFA_SIMULATION_CORE_API CollisionResetVisitor : public CollisionVisitor
{

public:
    CollisionResetVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj) override;
    const char* getClassName() const override { return "CollisionResetVisitor"; }
};

/// Compute collision detection
class SOFA_SIMULATION_CORE_API CollisionDetectionVisitor : public CollisionVisitor
{
public:
    CollisionDetectionVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj) override;
    const char* getClassName() const override { return "CollisionDetectionVisitor"; }
};

/// Compute collision response
class SOFA_SIMULATION_CORE_API CollisionResponseVisitor : public CollisionVisitor
{
public:
    CollisionResponseVisitor(const core::ExecParams* params) :CollisionVisitor(params) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj) override;
    const char* getClassName() const override { return "CollisionResponseVisitor"; }
};


} // namespace sofa::simulation


#endif
