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

/** Get vector size */
class SOFA_SIMULATION_CORE_API MechanicalVSizeVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::ConstMultiVecId v;
    size_t* result;

    MechanicalVSizeVisitor(const sofa::core::ExecParams* eparams, size_t* resultPtr, sofa::core::ConstMultiVecId vecid)
            : BaseMechanicalVisitor(eparams), v(vecid), result(resultPtr)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdMechanicalState(simulation::Node*,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node*,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVSizeVisitor";}
    std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(v);
    }
#endif
};
}
