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

/** Perform a vector operation v=a+b*f
*/
class SOFA_SIMULATION_CORE_API MechanicalVOpVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::MultiVecId v;
    sofa::core::ConstMultiVecId a;
    sofa::core::ConstMultiVecId b;
    SReal f;
    bool mapped;
    bool only_mapped;
    MechanicalVOpVisitor(const sofa::core::ExecParams* params,
                         sofa::core::MultiVecId v,sofa::core::ConstMultiVecId a = sofa::core::ConstMultiVecId::null(), sofa::core::ConstMultiVecId b = sofa::core::ConstMultiVecId::null(),
                         SReal f=1.0 )
            : BaseMechanicalVisitor(params) , v(v), a(a), b(b), f(f), mapped(false), only_mapped(false)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    // If mapped or only_mapped is ste, this visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    MechanicalVOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }
    MechanicalVOpVisitor& setOnlyMapped(bool m = true) { only_mapped = m; return *this; }

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    const char* getClassName() const override { return "MechanicalVOpVisitor";}
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
        addWriteVector(v);
    }
#endif
};
}