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

namespace sofa::simulation::mechanicalvisitor
{

/** Compute the dot product of two vectors */
class SOFA_SIMULATION_CORE_API MechanicalVDotVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::ConstMultiVecId a;
    sofa::core::ConstMultiVecId b;
    SReal* const m_total { nullptr };

    MechanicalVDotVisitor(const sofa::core::ExecParams* eparams, sofa::core::ConstMultiVecId avecid, sofa::core::ConstMultiVecId bvecid, SReal* t)
            : BaseMechanicalVisitor(eparams) , a(avecid), b(bvecid), m_total(t)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVDotVisitor";}
    std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(a);
        addReadVector(b);
    }
#endif
};
}
