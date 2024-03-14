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
#include <sofa/core/behavior/BaseAnimationLoop.h>

#include <sofa/simulation/fwd.h>


namespace sofa::core
{
class ExecParams;
}


namespace sofa::simulation
{

/**
 *  \brief Default Animation Loop to be created when no AnimationLoop found on simulation::node.
 *
 *
 */
class SOFA_SIMULATION_CORE_API DefaultAnimationLoop : public sofa::core::behavior::BaseAnimationLoop
{
public:
    typedef sofa::core::behavior::BaseAnimationLoop Inherit;
    typedef sofa::core::objectmodel::BaseContext BaseContext;
    typedef sofa::core::objectmodel::BaseObjectDescription BaseObjectDescription;
    SOFA_CLASS(DefaultAnimationLoop, sofa::core::behavior::BaseAnimationLoop);
protected:
    explicit DefaultAnimationLoop(simulation::Node* gnode = nullptr);

    ~DefaultAnimationLoop() override;

public:
    Data<bool> d_parallelODESolving; ///<If true, solves ODE solvers in parallel

    void init() override;

    /// Set the simulation node this animation loop is controlling
    virtual void setNode(simulation::Node*);

    /// perform one animation step
    void step(const sofa::core::ExecParams* params, SReal dt) override;

protected :
    simulation::Node* m_node { nullptr };

    void behaviorUpdatePosition(const sofa::core::ExecParams* params, SReal dt) const;
    void updateInternalData(const sofa::core::ExecParams* params) const;
    void beginIntegration(const sofa::core::ExecParams* params, SReal dt) const;
    void propagateIntegrateBeginEvent(const sofa::core::ExecParams* params) const;
    void accumulateMatrixDeriv(sofa::core::ConstraintParams cparams) const;
    void solve(const sofa::core::ExecParams* params, SReal dt) const;
    void propagateIntegrateEndEvent(const sofa::core::ExecParams* params) const;
    void endIntegration(const sofa::core::ExecParams* params, SReal dt) const;
    void projectPositionAndVelocity(SReal nextTime, const sofa::core::MechanicalParams& mparams) const;
    void propagateOnlyPositionAndVelocity(SReal nextTime, const sofa::core::MechanicalParams& mparams) const;
    void propagateCollisionBeginEvent(const sofa::core::ExecParams* params) const;
    void propagateCollisionEndEvent(const sofa::core::ExecParams* params) const;
    void collisionDetection(const sofa::core::ExecParams* params) const;
    void animate(const sofa::core::ExecParams* params, SReal dt) const;
    void updateSimulationContext(const sofa::core::ExecParams* params, SReal dt, SReal startTime) const;
    void propagateAnimateEndEvent(const sofa::core::ExecParams* params, SReal dt) const;
    void updateMapping(const sofa::core::ExecParams* params, SReal dt) const;
    void computeBoundingBox(const sofa::core::ExecParams* params) const;
    void propagateAnimateBeginEvent(const sofa::core::ExecParams* params, SReal dt) const;

};

} // namespace sofa::simulation
