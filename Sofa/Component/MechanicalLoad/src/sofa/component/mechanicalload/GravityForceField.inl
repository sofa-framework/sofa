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

#include "GravityForceField.h"
#include <sofa/core/MechanicalParams.h>

namespace sofa::component::mechanicalload
{


using namespace sofa::type;


template<class DataTypes>
GravityForceField<DataTypes>::GravityForceField()
    : d_gravitationalAcceleration(initData(&d_gravitationalAcceleration, "gravitationalAcceleration", "Value corresponding to the gravitational acceleration"))
    , l_mass(initLink("mass", "link to the mass"))
{
}


template<class DataTypes>
void GravityForceField<DataTypes>::init()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
       
    if (l_mass.empty())
    {
        msg_info() << "link to the mass should be set to ensure right behavior. First mass found in current context will be used.";
        sofa::core::behavior::Mass<DataTypes>* p_mass;
        this->getContext()->get(p_mass);
        l_mass.set(p_mass);
    }

    // temporary pointer to the mass
    if (sofa::core::behavior::Mass<DataTypes>* _mass = l_mass.get())
    {
        msg_info() << "Mass path used: '" << l_mass.getLinkedPath() << "'";
    }
    else
    {
        msg_error() << "No mass component found at path: " << l_mass.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        return;
    }

    // init from ForceField
    Inherit::init();

    // if all init passes, component is valid
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template<class DataTypes>
void GravityForceField<DataTypes>::setGravitationalAcceleration(const DPos grav)
{
    d_gravitationalAcceleration.setValue(grav);
}


template<class DataTypes>
void GravityForceField<DataTypes>::addForce(const core::MechanicalParams* params, DataVecDeriv& f, const DataVecCoord& x1, const DataVecDeriv& v1)
{
    sofa::core::behavior::Mass<DataTypes>* _mass = l_mass.get();
    Deriv gravity;
    DataTypes::setDPos(gravity, d_gravitationalAcceleration.getValue());
    _mass->addGravitationalForce(params,f,x1,v1,gravity);
}


template <class DataTypes>
SReal GravityForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* params, const DataVecCoord& x) const
{
    sofa::core::behavior::Mass<DataTypes>* _mass = l_mass.get();
    Deriv gravity;
    DataTypes::setDPos(gravity, d_gravitationalAcceleration.getValue());
    if(_mass)
        return _mass->getGravitationalPotentialEnergy(params, x, gravity);
    else
        return 0.0;
}


template<class DataTypes>
void GravityForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx)
{
    // Derivative of a constant gravity field is null, no need to compute addKToMatrix nor addDForce
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(d_df);
    SOFA_UNUSED(d_dx);
    mparams->setKFactorUsed(true);
}


template<class DataTypes>
void GravityForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal k, unsigned int & offset)
{
    // Derivative of a constant gravity field is null, no need to compute addKToMatrix nor addDForce
    SOFA_UNUSED(mat);
    SOFA_UNUSED(k);
    SOFA_UNUSED(offset);
}



} // namespace sofa::component::mechanicalload
