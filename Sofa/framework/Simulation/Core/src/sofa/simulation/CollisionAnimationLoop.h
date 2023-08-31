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
#ifndef SOFA_SIMULATION_TREE_COLLISIONANIMATIONLOOP_H
#define SOFA_SIMULATION_TREE_COLLISIONANIMATIONLOOP_H

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>

#include <sofa/simulation/config.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{


/**
 *  \brief Component responsible for main simulation algorithms, managing how
 *  and when collisions and integrations computations happen.
 *
 *  This class can optionally replace the default computation scheme of computing
 *  collisions then doing an integration step.
 *
 *  Note that it is in a preliminary stage, hence its fonctionnalities and API will
 *  certainly change soon.
 *
 */


class SOFA_SIMULATION_CORE_API CollisionAnimationLoop : public sofa::core::behavior::BaseAnimationLoop
{
public:
    typedef sofa::core::behavior::BaseAnimationLoop Inherit;
    typedef sofa::core::objectmodel::BaseContext BaseContext;
    typedef sofa::core::objectmodel::BaseObjectDescription BaseObjectDescription;

protected:
    CollisionAnimationLoop();
    ~CollisionAnimationLoop() override;

protected:

    /// @name Visitors
    /// These methods provides an abstract view of the mechanical system to animate.
    /// They are implemented by executing Visitors in the subtree of the scene-graph below this solver.
    /// @{

    /// Function meant to be called before the actual collision computation
    virtual void preCollisionComputation(const core::ExecParams* params = core::execparams::defaultInstance());
    /// Function performing the actual collision computation
    virtual void internalCollisionComputation(const core::ExecParams* params = core::execparams::defaultInstance());
    /// Function meant to be called after the actual collision computation
    virtual void postCollisionComputation(const core::ExecParams* params = core::execparams::defaultInstance());

    /// Activate collision pipeline
    virtual void computeCollision(const core::ExecParams* params = core::execparams::defaultInstance());

    /// Activate OdeSolvers
    virtual void integrate(const core::ExecParams* params, SReal dt);


    typedef simulation::Node::Sequence<core::behavior::OdeSolver> Solvers;
    typedef core::collision::Pipeline Pipeline;
    const Solvers& getSolverSequence();
};

} // namespace simulation

} // namespace sofa

#endif /* SOFA_SIMULATION_TREE_COLLISIONANIMATIONLOOP_H */
