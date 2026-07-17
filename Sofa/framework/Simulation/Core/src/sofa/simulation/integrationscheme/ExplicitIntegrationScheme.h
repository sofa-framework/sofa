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
#include <sofa/simulation/config.h>

#include <sofa/core/behavior/BaseIntegrationScheme.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MultiVec.h>

#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/simulation/MappingGraphMechanicalOperations.h>

namespace sofa::simulation::common
{
class MechanicalOperations;
class VectorOperations;
}

namespace sofa::simulation::integrationscheme
{

class SOFA_SIMULATION_CORE_API ExplicitIntegrationScheme :
                            public  sofa::core::behavior::BaseIntegrationScheme
{
public:
    SOFA_ABSTRACT_CLASS(ExplicitIntegrationScheme, sofa::core::behavior::BaseIntegrationScheme);

    ExplicitIntegrationScheme() = default;
    virtual ~ExplicitIntegrationScheme() override = default ;

    virtual void integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override final;
    virtual void doIntegrate(const core::ExecParams* params, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) = 0 ;

protected:
    std::shared_ptr<sofa::simulation::common::VectorOperations > m_vop;
    std::unique_ptr<sofa::simulation::common::MappingGraphMechanicalOperations> m_mop;

};
} // namespace sofa::component::integrationscheme


