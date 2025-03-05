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

/** Count the number of masses which are not diagonal */
class SOFA_SIMULATION_CORE_API MechanicalGetNonDiagonalMassesCountVisitor : public MechanicalVisitor
{
public:
    sofa::Size* const m_nbNonDiagonalMassesPtr { nullptr };

    // SOFA_ATTRIBUTE_DISABLED("v24.06", "v24.12", "given result is not a Real anymore since https://github.com/sofa-framework/sofa/pull/4328")
    MechanicalGetNonDiagonalMassesCountVisitor(const sofa::core::MechanicalParams* mparams, SReal* result) = delete;

    MechanicalGetNonDiagonalMassesCountVisitor(const sofa::core::MechanicalParams* mparams, sofa::Size* result)
        : MechanicalVisitor(mparams), m_nbNonDiagonalMassesPtr(result)
    {
    }

    Result fwdMass(VisitorContext* ctx, sofa::core::behavior::BaseMass* mass) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalGetNonDiagonalMassesCountVisitor";}
};

}
