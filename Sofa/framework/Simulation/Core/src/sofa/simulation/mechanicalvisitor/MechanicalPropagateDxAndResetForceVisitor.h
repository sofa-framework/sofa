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

/** Same as MechanicalPropagateDxVisitor followed by MechanicalResetForceVisitor
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateDxAndResetForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId dx,f;

    MechanicalPropagateDxAndResetForceVisitor(const sofa::core::MechanicalParams* mechaparams,
                                              sofa::core::MultiVecDerivId dxvecid, sofa::core::MultiVecDerivId fvecid)
            : MechanicalVisitor(mechaparams) , dx(dxvecid), f(fvecid)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateDxAndResetForceVisitor";}
    std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(dx);
        addWriteVector(f);
    }
#endif
};

} // namespace sofa::simulation::mechanicalvisitor
