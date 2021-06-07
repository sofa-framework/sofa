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

#include <sofa/core/ConstraintParams.h>

namespace sofa::simulation::mechanicalvisitor
{

class SOFA_SIMULATION_CORE_API MechanicalResetConstraintVisitor : public BaseMechanicalVisitor
{
public:
    //VecId res;
    MechanicalResetConstraintVisitor(const sofa::core::ConstraintParams* cparams)
            : BaseMechanicalVisitor(cparams)
            , m_cparams(cparams)
    {}

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* mm) override;

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalResetConstraintVisitor"; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }

private:
    const sofa::core::ConstraintParams* m_cparams;
};
}