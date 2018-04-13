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
#ifndef SOFA_SMP_PARALLELSOLVERIMPL_H
#define SOFA_SMP_PARALLELSOLVERIMPL_H

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>
#ifdef SOFA_SMP
#include <sofa/core/behavior/ParallelMultivector.h>
#endif
namespace sofa
{

namespace simulation
{
namespace common
{


/**
 *  \brief Implementation of LinearSolver/OdeSolver/AnimationLoop relying on GNode.
 *
 */
class ParallelSolverImpl : public virtual sofa::simulation::SolverImpl
{
public:
    ParallelSolverImpl();

    virtual ~ParallelSolverImpl();

    /// @name Visitors and MultiVectors
    /// These methods provides an abstract view of the mechanical system to animate.
    /// They are implemented by executing Visitors in the subtree of the scene-graph below this solver.
    /// @{

    /// @name Vector operations
    /// Most of these operations can be hidden by using the MultiVector class.
    /// @{

    /// Wait for the completion of previous operations and return the result of the last v_dot call.
    ///
    /// Note that currently all methods are blocking so finish simply return the result of the last v_dot call.


    virtual void v_op(core::VecId v, core::VecId a, core::VecId b, Shared<double> &f); ///< v=a+b*f

    virtual void v_dot(sofa::defaulttype::Shared<double> &result,core::VecId a, core::VecId b); ///< a dot b
    virtual void v_peq(core::VecId v, core::VecId a, Shared<double> &fSh, double f=1.0); ///< v+=f*a
    virtual void v_peq(core::VecId v, core::VecId a, double f=1.0); ///< v+=f*a

    virtual void v_meq(core::VecId v, core::VecId a, Shared<double> &fSh); ///< v+=f*a




};



} // namespace simulation
} // namespace simulation

} // namespace sofa

#endif
