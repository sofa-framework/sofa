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

/** Accumulate the product of the mass matrix by a given vector.
Typically used in implicit integration solved by a Conjugate Gradient algorithm.
Note that if a dx vector is given, it is used and propagated by the mappings, Otherwise the current value is used.
*/
class SOFA_SIMULATION_CORE_API MechanicalAddMDxVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    sofa::core::MultiVecDerivId dx;
    SReal factor;
    MechanicalAddMDxVisitor(const sofa::core::MechanicalParams* mechaparams,
                            sofa::core::MultiVecDerivId resvecid, sofa::core::MultiVecDerivId dxvecid, SReal f)
            : MechanicalVisitor(mechaparams), res(resvecid), dx(dxvecid), factor(f)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMass(simulation::Node* /*node*/,sofa::core::behavior::BaseMass* mass) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAddMDxVisitor"; }
    virtual std::string getInfos() const override { std::string name="dx["+dx.getName()+"] in res[" + res.getName()+"]"; return name; }

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(res);
    }
#endif
};
}
