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

/// Call each BaseConstraintSet to build the Jacobian matrices and accumulate it through the mappings up to the independant DOFs
/// @deprecated use MechanicalBuildConstraintMatrix followed by MechanicalAccumulateMatrixDeriv
SOFA_ATTRIBUTE_DISABLED_MECHANICALACCUMULATECONSTRAINT()
class SOFA_SIMULATION_CORE_API MechanicalAccumulateConstraint : public BaseMechanicalVisitor
{
public:
    MechanicalAccumulateConstraint(const sofa::core::ConstraintParams* _cparams,
                                   sofa::core::MultiMatrixDerivId _res, unsigned int &_contactId)
            : BaseMechanicalVisitor(_cparams)
            , res(_res)
            , contactId(_contactId)
            , cparams(_cparams)
    {}

    const sofa::core::ConstraintParams* constraintParams() const { return cparams; }

    Result fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* c) override;

    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAccumulateConstraint"; }

    bool isThreadSafe() const override
    {
        return false;
    }

protected:
    sofa::core::MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};
}
