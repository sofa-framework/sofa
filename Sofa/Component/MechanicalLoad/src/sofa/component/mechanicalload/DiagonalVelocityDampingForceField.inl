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
    : dampingCoefficients(initData(&dampingCoefficients, "dampingCoefficient", "velocity damping coefficients (by cinematic dof)"))
{
}



template<class DataTypes>
void DiagonalVelocityDampingForceField<DataTypes>::addForce(const core::MechanicalParams*, DataVecDeriv&_f, const DataVecCoord&, const DataVecDeriv&_v)
{
    const auto coefs = sofa::helper::getReadAccessor(dampingCoefficients);

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
void DiagonalVelocityDampingForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    unsigned nbDampingCoeff = dampingCoefficients.getValue().size();
    Real bfactor = (Real)mparams->bFactor();

    if( nbDampingCoeff && bfactor )
    {
        sofa::helper::WriteAccessor<DataVecDeriv> df(d_df);
        const VecDeriv& dx = d_dx.getValue();

        for(unsigned i=0; i<dx.size();i++)
            for(unsigned j=0; j<Deriv::total_size; j++)
                if( i<nbDampingCoeff )
                    df[i][j] -= dx[i][j]*dampingCoefficients.getValue()[i][j]*bfactor;
                else
                    df[i][j] -= dx[i][j]*dampingCoefficients.getValue().back()[j]*bfactor;
    }
}

template<class DataTypes>
void DiagonalVelocityDampingForceField<DataTypes>::addBToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal bFact, unsigned int& offset)
{
    const unsigned int size = this->mstate->getSize();
    unsigned nbDampingCoeff = dampingCoefficients.getValue().size();

    if( !nbDampingCoeff ) return;

    for( unsigned i=0 ; i<size ; i++ )
    {
        unsigned blockrow = offset+i*Deriv::total_size;
        for( unsigned j=0 ; j<Deriv::total_size ; j++ )
        {
            unsigned row = blockrow+j;
            if( i<nbDampingCoeff )
                mat->add( row, row, -dampingCoefficients.getValue()[i][j]*bFact );
            else
                mat->add( row, row, -dampingCoefficients.getValue().back()[j]*bFact );
        }
    }
}

template <class DataTypes>
SReal DiagonalVelocityDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    return 0;
}

} // namespace sofa::component::mechanicalload
