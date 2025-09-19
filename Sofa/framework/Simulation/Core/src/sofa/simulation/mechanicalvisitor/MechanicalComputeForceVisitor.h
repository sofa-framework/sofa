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

/** Accumulate the forces (internal and interactions).
This action is typically called after a MechanicalResetForceVisitor.
*/
class SOFA_SIMULATION_CORE_API MechanicalComputeForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings

    MechanicalComputeForceVisitor(const sofa::core::MechanicalParams* mechaparams,
                                  sofa::core::MultiVecDerivId resvecid, bool bAccumulate = true )
            : MechanicalVisitor(mechaparams) , res(resvecid), accumulate(bAccumulate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdForceField(simulation::Node* /*node*/,sofa::core::behavior::BaseForceField* ff) override;
    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override {return "MechanicalComputeForceVisitor";}
    std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addWriteVector(res);
    }
#endif
};


}
