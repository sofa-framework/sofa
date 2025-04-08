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
    : d_dampingCoefficientVector(initData(&d_dampingCoefficientVector, "dampingCoefficientVector", "Vector of velocity damping coefficients (by cinematic dof and by node)"))
    , d_dampingCoefficient(initData(&d_dampingCoefficient, "dampingCoefficient", "Node-constant and isotropic damping coefficient"))
{
    sofa::core::objectmodel::Base::addUpdateCallback("updateFromDampingCoefficientVector", {&d_dampingCoefficientVector}, [this](const core::DataTracker& )
    {
        msg_info() << "call back update: from dampingCoefficientVector";
        return updateFromDampingCoefficientVector();
    }, {});

    sofa::core::objectmodel::Base::addUpdateCallback("updateFromSingleDampingCoefficient", {&d_dampingCoefficient}, [this](const core::DataTracker& )
    {
        msg_info() << "call back update: from dampingCoefficient";
        return updateFromSingleDampingCoefficient();
    }, {});
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState NodalLinearDampingForceField<DataTypes>::updateFromDampingCoefficientVector()
{
    // Check if the dampingCoefficientVector has a null size
    VecDeriv dampingCoefficients = d_dampingCoefficientVector.getValue();
    const unsigned int sizeCoefs = dampingCoefficients.size();
    if(sizeCoefs == 0)
    {
        msg_error() << "Size of the \'dampingCoefficientVector\' vector is null";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }

    // Recover mstate size
    const unsigned int sizeMState = this->mstate->getSize();


    // If the sizes of the dampingCoefficientVector and the mechanical state mismatch
    if(sizeCoefs != sizeMState && sizeCoefs != 1)
    {
        msg_error() << "Size of the \'dampingCoefficientVector\' vector does not fit the associated MechanicalObject size";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }


    // If size=1, make a constant vector which size fits the MechanicalObject
    if(sizeCoefs == 1)
    {
        sofa::helper::WriteAccessor<Data<VecDeriv> > accessorCoefficient = this->d_dampingCoefficientVector;
        Deriv constantCoef = accessorCoefficient[0];
        accessorCoefficient.resize(sizeMState);

        for(unsigned j = 0; j < Deriv::total_size; ++j)
        {
            if(constantCoef[j] < 0.)
            {
                msg_error() << "Negative entry given in the constant \'dampingCoefficientVector\'";
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
                if(dampingCoefficients[i][j] < 0.)
                {
                    msg_error() << "Negative value at the " << i << "th entry of the \'dampingCoefficientVector\' vector";
                    return sofa::core::objectmodel::ComponentState::Invalid;
                }
            }
        }
    }

    msg_info() << "Update from dampingCoefficientVector successfully done";
    isConstantIsotropic = false;
    return sofa::core::objectmodel::ComponentState::Valid;
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState NodalLinearDampingForceField<DataTypes>::updateFromSingleDampingCoefficient()
{
    // Check if the given dampingCoefficient
    Real dampingCoefficient = d_dampingCoefficient.getValue();
    
    if(dampingCoefficient < 0.)
    {
        msg_error() << "Negative \'dampingCoefficient\' given";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }

    isConstantIsotropic = true;
    msg_info() << "Update from dampingCoefficient successfully done";
    return sofa::core::objectmodel::ComponentState::Valid;
}


template<class DataTypes>
void NodalLinearDampingForceField<DataTypes>::init()
{
    Inherit::init();

    // Case no input is given
    if(!d_dampingCoefficient.isSet() && !d_dampingCoefficientVector.isSet())
    {
        msg_error() << "One damping coefficient should be specified (either \'dampingCoefficientVector\' or \'dampingCoefficient\')";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // Too many input damping information
    if(d_dampingCoefficient.isSet() && d_dampingCoefficientVector.isSet())
    {
        msg_warning() << "Too many input damping information: either using \'dampingCoefficientVector\' or \'dampingCoefficient\' should be specified \ndampingCoefficient will be used";
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template<class DataTypes>
void NodalLinearDampingForceField<DataTypes>::addForce(const core::MechanicalParams*, DataVecDeriv&_f, const DataVecCoord& _x, const DataVecDeriv&_v)
{
    if(!this->isComponentStateValid())
        return;

    SOFA_UNUSED(_x);

    sofa::helper::WriteAccessor<DataVecDeriv> f(_f);
    const VecDeriv& v = _v.getValue();

    // Constant and isotropic damping
    if(isConstantIsotropic)
    {
        const Real singleCoefficient = d_dampingCoefficient.getValue();

        for(std::size_t i = 0; i < v.size(); ++i)
        {
f[i] -= v[i] * singleCoefficient;
        }
    }
    else
    {
        const auto coefs = sofa::helper::getReadAccessor(d_dampingCoefficientVector);

        if( !coefs.empty() )
        {
            const std::size_t nbDampingCoeff = coefs.size();

            for(std::size_t i = 0; i < v.size(); ++i)
            {
                const auto coefId = std::min(i, nbDampingCoeff-1);
f[i] -= v[i] * coefs[coefId];
            }
        }
    }
}

template<class DataTypes>
void NodalLinearDampingForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    if(!this->isComponentStateValid())
        return;
    
    sofa::helper::WriteAccessor<DataVecDeriv> df(d_df);
    const VecDeriv& dx = d_dx.getValue();

    const Real bfactor = (Real)mparams->bFactor();

    // Constant and isotropic damping
    if(isConstantIsotropic)
    {
        const Real singleCoefficient = d_dampingCoefficient.getValue();
        Real factor = singleCoefficient * bfactor;

        for (std::size_t i = 0; i < dx.size(); i++)
        {
df[i] -= dx[i] * factor;
        }
    }
    else
    {
        const auto& coefs = d_dampingCoefficientVector.getValue();
        std::size_t nbDampingCoeff = coefs.size();

        if (nbDampingCoeff && bfactor)
        {

            for (std::size_t i = 0; i < dx.size(); i++)
            {
df[i] -= dx[i] * coefs[i[ * bfactor;
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
    if(!this->isComponentStateValid())
        return;
    
    // Constant and isotropic damping
    if(isConstantIsotropic && bFact)
    {
        const Real factor = d_dampingCoefficient.getValue() * bFact;
        const unsigned int size = this->mstate->getSize();

        for (std::size_t i = 0; i < size; i++)
        {
            const unsigned blockrow = offset + i * Deriv::total_size;
            for (unsigned j = 0; j < Deriv::total_size; j++)
            {
                unsigned row = blockrow + j;
                mat->add(row, row, -factor);
            }
        }
    }
    else
    {
        const auto& coefs = d_dampingCoefficientVector.getValue();
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
}

template <class DataTypes>
void NodalLinearDampingForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix* matrix)
{
    if(!this->isComponentStateValid())
        return;
    
    auto dfdv = matrix->getForceDerivativeIn(this->mstate)
        .withRespectToVelocityIn(this->mstate);

    const unsigned int size = this->mstate->getSize();

    // Constant and isotropic damping
    if(isConstantIsotropic)
    {
        const Real dampingCoefficient = d_dampingCoefficient.getValue();

        for (std::size_t i = 0; i < size; i++)
        {
            const unsigned blockrow = i * Deriv::total_size;
            for (unsigned j = 0; j < Deriv::total_size; j++)
            {
                const unsigned row = blockrow + j;
                dfdv(row, row) += -dampingCoefficient;
            }
        }
    }
    else
    {
        const auto& coefs = d_dampingCoefficientVector.getValue();
        const std::size_t nbDampingCoeff = coefs.size();

        if (nbDampingCoeff)
        {
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
}

template <class DataTypes>
SReal NodalLinearDampingForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    return 0;
}

} // namespace sofa::component::mechanicalload
