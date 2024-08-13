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

#include <sofa/component/mechanicalload/NodalLinearDampingForceField.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

namespace sofa::component::mechanicalload
{


template<class DataTypes>
NodalLinearDampingForceField<DataTypes>::NodalLinearDampingForceField()
    : d_dampingCoefficients(initData(&d_dampingCoefficients, "dampingCoefficient", "Velocity damping coefficients (by cinematic dof)"))
{
    sofa::core::objectmodel::Base::addUpdateCallback("updateFromDampingCoefficient", {&d_dampingCoefficients}, [this](const core::DataTracker& )
    {
        msg_info() << "call back update: from dampingCoefficient";
        return updateFromDampingCoefficient();
    }, {});
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState NodalLinearDampingForceField<DataTypes>::updateFromDampingCoefficient()
{
    const auto coefs = sofa::helper::getReadAccessor(d_dampingCoefficients);
    const unsigned int sizeCoefs = coefs.size();


    // Check if the dampingCoefficients vector has a null size
    if(sizeCoefs == 0)
    {
        msg_error() << "Size of the \'dampingCoefficients\' vector is null";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }

    // Recover mstate size
    const unsigned int sizeMState = this->mstate->getSize();


    // If the sizes of the dampingCoefficients vector and the mechanical state mismatch
    if(sizeCoefs != sizeMState && sizeCoefs != 1)
    {
        msg_error() << "Size of the \'dampingCoefficients\' vector does not fit the associated MechanicalObject size";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }


    // If size=1, make a constant vector which size fits the MechanicalObject
    if(sizeCoefs == 1)
    {
        sofa::helper::WriteAccessor<Data<VecDeriv> > accessorCoefficient = this->d_dampingCoefficients;
        Deriv constantCoef = accessorCoefficient[0];
        accessorCoefficient.resize(sizeMState);

        for(unsigned j = 0; j < Deriv::total_size; ++j)
        {
            if(constantCoef[j] < 0.)
            {
                msg_error() << "Negative \'dampingCoefficients\' given";
                return sofa::core::objectmodel::ComponentState::Invalid;
            }
        }

        for (unsigned i = 0; i < sizeMState; ++i)
        {
            accessorCoefficient[i] = constantCoef;
        }
    }
    else
    {
        for (unsigned i = 0; i < sizeMState; ++i)
        {
            for(unsigned j = 0; j < Deriv::total_size; ++j)
            {
                if(coefs[i][j] < 0.)
                {
                    msg_error() << "Negative value at the " << i << "th entry of the \'dampingCoefficients\' vector";
                    return sofa::core::objectmodel::ComponentState::Invalid;
                }
            }
        }
    }

    msg_info() << "Update from dampingCoefficient successfully done";
    return sofa::core::objectmodel::ComponentState::Valid;
}


template<class DataTypes>
void NodalLinearDampingForceField<DataTypes>::addForce(const core::MechanicalParams*, DataVecDeriv&_f, const DataVecCoord& _x, const DataVecDeriv&_v)
{
    SOFA_UNUSED(_x);
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
void NodalLinearDampingForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    const auto& coefs = d_dampingCoefficients.getValue();
    std::size_t nbDampingCoeff = coefs.size();
    const Real bfactor = (Real)mparams->bFactor();

    if (nbDampingCoeff && bfactor)
    {
        sofa::helper::WriteAccessor<DataVecDeriv> df(d_df);
        const VecDeriv& dx = d_dx.getValue();

        for (std::size_t i = 0; i < dx.size(); i++)
        {
            for (unsigned j = 0; j < Deriv::total_size; j++)
            {
                df[i][j] -= dx[i][j] * coefs[i][j] * bfactor;
            }
        }
    }
}

template <class DataTypes>
void NodalLinearDampingForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix*)
{
    // NodalLinearDampingForceField is a pure damping component
    // No stiffness is computed
}

template<class DataTypes>
void NodalLinearDampingForceField<DataTypes>::addBToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal bFact, unsigned int& offset)
{
    const auto& coefs = d_dampingCoefficients.getValue();
    const std::size_t nbDampingCoeff = coefs.size();

    if (nbDampingCoeff && bFact)
    {
        const unsigned int size = this->mstate->getSize();
        for (std::size_t i = 0; i < size; i++)
        {
            const unsigned blockrow = offset + i * Deriv::total_size;
            for (unsigned j = 0; j < Deriv::total_size; j++)
            {
                unsigned row = blockrow + j;
                mat->add(row, row, -coefs[i][j] * bFact);
            }
        }
    }
}

template <class DataTypes>
void NodalLinearDampingForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix* matrix)
{
    const auto& coefs = d_dampingCoefficients.getValue();
    const std::size_t nbDampingCoeff = coefs.size();

    if (nbDampingCoeff)
    {
        auto dfdv = matrix->getForceDerivativeIn(this->mstate)
                           .withRespectToVelocityIn(this->mstate);

        const unsigned int size = this->mstate->getSize();
        for (std::size_t i = 0; i < size; i++)
        {
            const unsigned blockrow = i * Deriv::total_size;
            for (unsigned j = 0; j < Deriv::total_size; j++)
            {
                const unsigned row = blockrow + j;
                dfdv(row, row) += -coefs[i][j];
            }
        }
    }
}

template <class DataTypes>
SReal NodalLinearDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    return 0;
}

} // namespace sofa::component::mechanicalload
