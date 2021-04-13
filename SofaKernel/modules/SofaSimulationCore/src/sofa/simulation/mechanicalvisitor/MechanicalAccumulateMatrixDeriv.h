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

/// Accumulate Jacobian matrices through the mappings up to the independant DOFs
class SOFA_SIMULATION_CORE_API MechanicalAccumulateMatrixDeriv : public BaseMechanicalVisitor
{
public:
    MechanicalAccumulateMatrixDeriv(const sofa::core::ConstraintParams* _cparams,
                                    sofa::core::MultiMatrixDerivId _res, bool _reverseOrder = false)
            : BaseMechanicalVisitor(_cparams)
            , res(_res)
            , cparams(_cparams)
            , reverseOrder(_reverseOrder)
    {}

    const sofa::core::ConstraintParams* constraintParams() const { return cparams; }

    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    /// Return true to reverse the order of traversal of child nodes
    bool childOrderReversed(simulation::Node* /*node*/) override { return reverseOrder; }

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAccumulateMatrixDeriv"; }

    bool isThreadSafe() const override
    {
        return false;
    }

protected:
    sofa::core::MultiMatrixDerivId res;
    const sofa::core::ConstraintParams *cparams;
    bool reverseOrder;
};
}