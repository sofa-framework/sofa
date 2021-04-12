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

/** Compute the total number of DOFs */
class SOFA_SIMULATION_CORE_API MechanicalGetDimensionVisitor : public MechanicalVisitor
{
public:
    MechanicalGetDimensionVisitor(const sofa::core::MechanicalParams* mparams, SReal* result)
            : MechanicalVisitor(mparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        rootData = result;
    }

    Result fwdMechanicalState(VisitorContext* ctx, sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalGetDimensionVisitor";}

    bool writeNodeData() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
    }
#endif
};

}