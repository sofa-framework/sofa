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

#include <sofa/component/mechanicalload/DiagonalVelocityDampingForceField.h>
#include <sofa/core/MechanicalParams.h>


namespace sofa::component::mechanicalload
{

template<class DataTypes>
DiagonalVelocityDampingForceField<DataTypes>::DiagonalVelocityDampingForceField()
    : d_dampingCoefficients(initData(&d_dampingCoefficients, "dampingCoefficient", "velocity damping coefficients (by cinematic dof)"))
{
}



template<class DataTypes>
void DiagonalVelocityDampingForceField<DataTypes>::addForce(const core::MechanicalParams*, DataVecDeriv&_f, const DataVecCoord&, const DataVecDeriv&_v)
{
    const auto coefs = sofa::helper::getReadAccessor(d_dampingCoefficients);

    if( !coefs.empty() )
    {
        const std::size_t nbDampingCoeff = coefs.size();

        sofa::helper::WriteAccessor<DataVecDeriv> f(_f);
        const VecDeriv& v = _v.getValue();

        for(std::size_t i = 0; i < v.size(); ++i)
        {
            const auto coefId = std::min(i, nbDampingCoeff-1);
            for(unsigned j = 0; j < Deriv::total_size; ++j)
            {
                f[i][j] -= v[i][j] * coefs[coefId][j];
            }
        }
    }
}

template<class DataTypes>
void DiagonalVelocityDampingForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /*d_df*/ , const DataVecDeriv& /*d_dx*/)
{
    (void)mparams->kFactor(); // get rid of warning message
}

template<class DataTypes>
void DiagonalVelocityDampingForceField<DataTypes>::addBToMatrix(sofa::linearalgebra::BaseMatrix * /*mat*/, SReal /*bFact*/, unsigned int& /*offset*/)
{
}

template <class DataTypes>
void DiagonalVelocityDampingForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template <class DataTypes>
SReal DiagonalVelocityDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    return 0;
}

} // namespace sofa::component::mechanicalload
