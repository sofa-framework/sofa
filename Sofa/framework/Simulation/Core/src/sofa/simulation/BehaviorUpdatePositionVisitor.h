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
#ifndef SOFA_SIMULATION_BEHAVIORUPDATEPOSITIONACTION_H
#define SOFA_SIMULATION_BEHAVIORUPDATEPOSITIONACTION_H


#include <sofa/simulation/Visitor.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/simulation/fwd.h>


namespace sofa::simulation
{

/** Update the position of a new simulation step

 */
class SOFA_SIMULATION_CORE_API BehaviorUpdatePositionVisitor : public Visitor
{

public:
    BehaviorUpdatePositionVisitor(const core::ExecParams* eparams, SReal _dt): Visitor(eparams),dt(_dt) {}
    void processBehaviorModel(simulation::Node* node, core::BehaviorModel* b);
    Result processNodeTopDown(simulation::Node* node) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "behavior update position"; }
    const char* getClassName() const override { return "BehaviorUpdatePositionVisitor"; }

    void setDt(SReal _dt) {dt = _dt;}
    SReal getDt() {return dt;}
protected:
    SReal dt;
};

} // namespace sofa::simulation


#endif
