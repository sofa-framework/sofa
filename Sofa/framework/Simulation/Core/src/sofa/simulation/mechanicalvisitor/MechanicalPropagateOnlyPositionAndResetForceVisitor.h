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

/** Same as MechanicalPropagateOnlyPositionVisitor followed by MechanicalResetForceVisitor

Note that this visitor only propagate through the mappings, and does
not apply projective constraints as was previously done by
MechanicalPropagatePositionAndResetForceVisitor.
Use MechanicalProjectPositionVisitor before this visitor if projection
is needed.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateOnlyPositionAndResetForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId f;

    MechanicalPropagateOnlyPositionAndResetForceVisitor(const sofa::core::MechanicalParams* mechaparams,
                                                        sofa::core::MultiVecCoordId xvecid, sofa::core::MultiVecDerivId fvecid)
            : MechanicalVisitor(mechaparams) , x(xvecid), f(fvecid)
    {
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateOnlyPositionAndResetForceVisitor"; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(x);
        addWriteVector(f);
    }
#endif
};

} // namespace sofa::simulation::mechanicalvisitor
