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


class SOFA_SIMULATION_CORE_API BaseCollisionVisitor : public Visitor
{
   public:
    BaseCollisionVisitor(const core::ExecParams* eparams) :Visitor(eparams) , m_primitiveTestCount(0) {}

    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);

    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "collision"; }
    const char* getClassName() const override { return "BaseCollisionVisitor"; }

    size_t getPrimitiveTestCount() const {return m_primitiveTestCount;}
   private:
    size_t m_primitiveTestCount;
};

class SOFA_SIMULATION_CORE_API ProcessGeometricalDataVisitor : public Visitor
{
   public:
    ProcessGeometricalDataVisitor(const core::ExecParams* eparams) :Visitor(eparams) {}

    virtual void fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet);
    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "collision"; }
    const char* getClassName() const override { return "ProcessGeometricalDataVisitor"; }

};

/// Compute collision reset, detection and response in one step
class SOFA_SIMULATION_CORE_API CollisionVisitor :  public BaseCollisionVisitor
{
public:
    CollisionVisitor(const core::ExecParams* eparams) : BaseCollisionVisitor(eparams) {}

    virtual void fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet);
    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "collision"; }
    const char* getClassName() const override { return "CollisionVisitor"; }

};

/// Remove collision response from last step
class SOFA_SIMULATION_CORE_API CollisionResetVisitor : public BaseCollisionVisitor
{

public:
    CollisionResetVisitor(const core::ExecParams* eparams) : BaseCollisionVisitor(eparams) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj) override;
    const char* getClassName() const override { return "CollisionResetVisitor"; }
};

/// Compute collision detection
class SOFA_SIMULATION_CORE_API CollisionDetectionVisitor : public BaseCollisionVisitor
{
public:
    CollisionDetectionVisitor(const core::ExecParams* eparams) : BaseCollisionVisitor(eparams) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj) override;
    const char* getClassName() const override { return "CollisionDetectionVisitor"; }
};

/// Compute collision response
class SOFA_SIMULATION_CORE_API CollisionResponseVisitor : public BaseCollisionVisitor
{
public:
    CollisionResponseVisitor(const core::ExecParams* eparams) : BaseCollisionVisitor(eparams) {}
    void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj) override;
    const char* getClassName() const override { return "CollisionResponseVisitor"; }
};


} // namespace sofa::simulation


#endif
