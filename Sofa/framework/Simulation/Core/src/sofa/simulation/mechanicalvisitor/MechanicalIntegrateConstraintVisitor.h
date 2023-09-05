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

/** Apply the motion correction computed from constraint force influence  */
class SOFA_SIMULATION_CORE_API MechanicalIntegrateConstraintsVisitor : public BaseMechanicalVisitor
{
public:
    const sofa::core::ConstraintParams* cparams;
    const double positionFactor;///< use the OdeSolver to get the position integration factor
    const double velocityFactor;///< use the OdeSolver to get the position integration factor
    sofa::core::ConstMultiVecDerivId correctionId;
    sofa::core::MultiVecDerivId dxId;
    sofa::core::MultiVecCoordId xId;
    sofa::core::MultiVecDerivId vId;
    int offset;

    MechanicalIntegrateConstraintsVisitor(
        const core::ConstraintParams* cparams,
        double pf, double vf,
        sofa::core::ConstMultiVecDerivId correction,
        sofa::core::MultiVecDerivId dx = sofa::core::MultiVecDerivId(sofa::core::VecDerivId::dx()),
        sofa::core::MultiVecCoordId x  = sofa::core::MultiVecCoordId(sofa::core::VecCoordId::position()),
        sofa::core::MultiVecDerivId v  = sofa::core::MultiVecDerivId(sofa::core::VecDerivId::velocity()));

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalIntegrateConstraintsVisitor"; }
};

} // namespace sofa::simulation::mechanicalvisitor
