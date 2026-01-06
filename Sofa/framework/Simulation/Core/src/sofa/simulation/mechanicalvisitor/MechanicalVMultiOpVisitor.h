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

#include <sofa/simulation/BaseMechanicalVisitor.h>

#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::simulation::mechanicalvisitor
{

/** Perform a sequence of linear vector accumulation operation $r_i = sum_j (v_j*f_{ij})
*
*  This is used to compute in on steps operations such as $v = v + a*dt, x = x + v*dt$.
*  Note that if the result vector appears inside the expression, it must be the first operand.
*/
class SOFA_SIMULATION_CORE_API MechanicalVMultiOpVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    bool mapped;
    MechanicalVMultiOpVisitor(const sofa::core::ExecParams* eparams, const VMultiOp& o)
            : BaseMechanicalVisitor(eparams), mapped(false), ops(o)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    MechanicalVMultiOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    const char* getClassName() const override { return "MechanicalVMultiOpVisitor"; }
    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        for (unsigned int i=0; i<ops.size(); ++i)
        {
            addWriteVector(ops[i].first);
            for (unsigned int j=0; j<ops[i].second.size(); ++j)
            {
                addReadVector(ops[i].second[j].first);
            }
        }
    }
#endif
    void setVMultiOp(VMultiOp &o)
    {
        ops = o;
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
protected:
    VMultiOp ops;
};
}
