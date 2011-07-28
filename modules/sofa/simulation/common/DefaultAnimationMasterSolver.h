/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_DEFAULTANIMATIONMASTERSOLVER_H
#define SOFA_SIMULATION_DEFAULTANIMATIONMASTERSOLVER_H

#include <sofa/core/behavior/MasterSolver.h>
#include <sofa/simulation/common/common.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/ExecParams.h>

namespace sofa
{

namespace simulation
{


/**
 *  \brief Default MasterSolver to be created when no Master Sovler found on simulation::node.
 *
 *
 */

class SOFA_SIMULATION_COMMON_API DefaultAnimationMasterSolver : public sofa::core::behavior::MasterSolver
{
public:
    SOFA_CLASS(DefaultAnimationMasterSolver,sofa::core::behavior::MasterSolver);

    DefaultAnimationMasterSolver();

    virtual ~DefaultAnimationMasterSolver();

    virtual void step(const core::ExecParams* params, double dt);


    virtual void computeCollision(const core::ExecParams* /*params*/) {}
    virtual void integrate(const core::ExecParams* /*params*/, double /*dt*/) {}

protected:
    typedef simulation::Node::Sequence<core::behavior::OdeSolver> Solvers;
    typedef core::collision::Pipeline Pipeline;
    const Solvers& getSolverSequence();
};

} // namespace simulation

} // namespace sofa

#endif  /* SOFA_SIMULATION_DEFAULTANIMATIONMASTERSOLVER_H */
