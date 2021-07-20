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
/**
* This class define a visitor which will go through the scene graph in reverse order and call the method applyJT of each mechanical mapping (@sa sofa::core::BaseMapping)
*/
class SOFA_SIMULATION_CORE_API MechanicalAccumulateJacobian : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalAccumulateJacobian(const core::ConstraintParams* _cparams, core::MultiMatrixDerivId _res);

    void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAccumulateJacobian"; }

    bool isThreadSafe() const override
    {
        return false;
    }
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

protected:
    core::MultiMatrixDerivId res;
    const sofa::core::ConstraintParams *cparams;
};

} //namespace sofa::simulation::mechanicalvisitor