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

#include <sofa/component/mechanicalload/UniformVelocityDampingForceField.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

namespace sofa::component::mechanicalload
{


template<class DataTypes>
UniformVelocityDampingForceField<DataTypes>::UniformVelocityDampingForceField()
    : dampingCoefficient(initData(&dampingCoefficient, Real(0.1), "dampingCoefficient", "velocity damping coefficient"))
    , d_implicit(initData(&d_implicit, false, "implicit", "should it generate damping matrix df/dv? (explicit otherwise, i.e. only generating a force)"))
{
    core::objectmodel::Base::addAlias( &dampingCoefficient, "damping" );
}

template<class DataTypes>
void UniformVelocityDampingForceField<DataTypes>::addForce (const core::MechanicalParams*, DataVecDeriv&_f, const DataVecCoord&, const DataVecDeriv&_v)
{
    sofa::helper::WriteAccessor<DataVecDeriv> f(_f);
    const VecDeriv& v = _v.getValue();

    for(unsigned int i=0; i<v.size(); i++)
        f[i] -= v[i]*dampingCoefficient.getValue();
}

template<class DataTypes>
void UniformVelocityDampingForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx)
{
    (void)mparams->kFactor(); // get rid of warning message

    if( !d_implicit.getValue() ) return;

    Real bfactor = (Real)sofa::core::mechanicalparams::bFactor(mparams);

    if( bfactor )
    {
        sofa::helper::WriteAccessor<DataVecDeriv> df(d_df);
        const VecDeriv& dx = d_dx.getValue();

        bfactor *= dampingCoefficient.getValue();

        for(unsigned int i=0; i<dx.size(); i++)
            df[i] -= dx[i]*bfactor;
    }
}

template<class DataTypes>
void UniformVelocityDampingForceField<DataTypes>::addBToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal bFact, unsigned int& offset)
{
    if( !d_implicit.getValue() ) return;

    const sofa::Size size = this->mstate->getMatrixSize();
    const auto dampingContribution = -dampingCoefficient.getValue() * bFact;

    for( sofa::Size i = 0 ; i < size; ++i )
    {
        mat->add( offset + i, offset + i, dampingContribution );
    }
}

template <class DataTypes>
void UniformVelocityDampingForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix* matrix)
{
    if( !d_implicit.getValue() ) return;

    auto dfdv = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToVelocityIn(this->mstate);

    const sofa::Size size = this->mstate->getMatrixSize();
    const auto damping = sofa::helper::ReadAccessor(dampingCoefficient);
    for( sofa::Size i = 0; i < size; ++i)
    {
        dfdv(i, i) += -damping.ref();
    }
}

template <class DataTypes>
SReal UniformVelocityDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    // TODO
    return 0;
}


} // namespace sofa::component::mechanicalload
