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
#include <sofa/component/constraint/lagrangian/solver/config.h>

#include <sofa/simulation/MechanicalVisitor.h>


namespace sofa::component::constraint::lagrangian::solver
{

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API ConstraintStoreLambdaVisitor : public simulation::BaseMechanicalVisitor
{
public:
    ConstraintStoreLambdaVisitor(const sofa::core::ConstraintParams* cParams, const sofa::linearalgebra::BaseVector* lambda);

    Visitor::Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet) override;

    void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

    bool stopAtMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

private:
    const sofa::core::ConstraintParams* m_cParams;
    const sofa::linearalgebra::BaseVector* m_lambda;
};


} // namespace sofa::component::constraint::lagrangian::solver
