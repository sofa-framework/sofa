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

#include <sofa/simulation/MechanicalVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

/** Propagate velocities to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.
After the execution of this action, all the (mapped) degrees of freedom are consistent with the independent degrees of freedom.

Note that this visitor only propagate through the mappings, and does
not apply projective constraints as was previously done by
MechanicalPropagateVelocityVisitor.
Use MechanicalProjectVelocityVisitor before this visitor if projection
is needed.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateOnlyVelocityVisitor : public MechanicalVisitor
{
public:
    SReal currentTime;
    sofa::core::MultiVecDerivId v;

    MechanicalPropagateOnlyVelocityVisitor(const sofa::core::MechanicalParams* mparams, SReal time=0,
                                           sofa::core::MultiVecDerivId v = sofa::core::VecId::velocity());

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateOnlyVelocityVisitor";}
    std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(v);
    }
#endif
};

} // namespace sofa::simulation::mechanicalvisitor
