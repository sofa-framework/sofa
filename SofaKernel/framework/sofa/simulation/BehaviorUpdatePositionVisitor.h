/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_SIMULATION_BEHAVIORUPDATEPOSITIONACTION_H
#define SOFA_SIMULATION_BEHAVIORUPDATEPOSITIONACTION_H


#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/ExecParams.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/BehaviorModel.h>

namespace sofa
{

namespace simulation
{

/** Update the position of a new simulation step

 */
class SOFA_SIMULATION_CORE_API BehaviorUpdatePositionVisitor : public Visitor
{

public:
    BehaviorUpdatePositionVisitor(const core::ExecParams* params, SReal _dt): Visitor(params),dt(_dt) {}
    void processBehaviorModel(simulation::Node* node, core::BehaviorModel* b);
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "behavior update position"; }
    virtual const char* getClassName() const { return "BehaviorUpdatePositionVisitor"; }

    void setDt(SReal _dt) {dt = _dt;}
    SReal getDt() {return dt;}
protected:
    SReal dt;
};

} // namespace simulation

} // namespace sofa

#endif
