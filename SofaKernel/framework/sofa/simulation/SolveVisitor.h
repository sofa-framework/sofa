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
#ifndef SOFA_SIMULATION_SOLVEACTION_H
#define SOFA_SIMULATION_SOLVEACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/Visitor.h>
#include <sofa/core/behavior/OdeSolver.h>

namespace sofa
{

namespace simulation
{

/** Used by the animation loop: send the solve signal to the others solvers

 */
class SOFA_SIMULATION_CORE_API SolveVisitor : public Visitor
{
public:
    SolveVisitor(const sofa::core::ExecParams* params, SReal _dt) : Visitor(params), dt(_dt), x(core::VecCoordId::position()),
        v(core::VecDerivId::velocity()) {}

    SolveVisitor(const sofa::core::ExecParams* params, SReal _dt, bool free) : Visitor(params), dt(_dt){
        if(free){
            x = core::VecCoordId::freePosition();
            v = core::VecDerivId::freeVelocity();
        }
        else{
            x = core::VecCoordId::position();
            v = core::VecDerivId::velocity();
        }
    }

    SolveVisitor(const sofa::core::ExecParams* params, SReal _dt, sofa::core::MultiVecCoordId X,sofa::core::MultiVecDerivId V) : Visitor(params), dt(_dt),
        x(X),v(V){}

    virtual void processSolver(simulation::Node* node, core::behavior::OdeSolver* b);
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "behavior update position"; }
    virtual const char* getClassName() const { return "SolveVisitor"; }

    void setDt(SReal _dt) {dt = _dt;}
    SReal getDt() {return dt;}
protected:
    SReal dt;
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId v;
};

} // namespace simulation

} // namespace sofa

#endif
