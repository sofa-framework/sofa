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
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.inl>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
StiffSpringForceField<DataTypes>::StiffSpringForceField(SReal ks, SReal kd)
    : StiffSpringForceField<DataTypes>(nullptr, nullptr, ks, kd)
{
}

template<class DataTypes>
StiffSpringForceField<DataTypes>::StiffSpringForceField(MechanicalState* object1, MechanicalState* object2, SReal ks, SReal kd)
    : SpringForceField<DataTypes>(object1, object2, ks, kd)
    , d_indices1(initData(&d_indices1, "indices1", "Indices of the source points on the first model"))
    , d_indices2(initData(&d_indices2, "indices2", "Indices of the fixed points on the second model"))
    , d_lengths(initData(&d_lengths, "lengths", "List of lengths to create the springs. Must have the same than indices1 & indices2, or if only one element, it will be applied to all springs. If empty, 0 will be applied everywhere"))
{
    this->addAlias(&d_lengths, "length");

    this->addUpdateCallback("updateSprings", { &d_indices1, &d_indices2, &d_lengths, &this->ks, &this->kd}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        createSpringsFromInputs();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {&this->springs});
}


template<class DataTypes>
void StiffSpringForceField<DataTypes>::init()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    if (d_indices1.isSet() && d_indices2.isSet())
    {
        createSpringsFromInputs();
    }
    this->SpringForceField<DataTypes>::init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template<class DataTypes>
void StiffSpringForceField<DataTypes>::createSpringsFromInputs()
{
    const auto& indices1 = d_indices1.getValue();
    const auto& indices2 = d_indices2.getValue();

    if (indices1.size() != indices2.size())
    {
        msg_error() << "Inputs indices sets sizes are different: d_indices1: " << indices1.size()
            << " | d_indices2 " << indices2.size()
            << " . No springs will be created";
        return;
    }

    if (indices1.empty())
        return;

    auto lengths = sofa::helper::getWriteAccessor(d_lengths);
    if (lengths.empty())
    {
        lengths.push_back({});
    }

    if (lengths.size() != indices1.size())
    {
        msg_warning() << "Lengths list has a different size than indices1. The list will be resized to " << indices1.size() << " elements.";
        lengths->resize(indices1.size(), lengths->back());
    }

    msg_info() << "Inputs have changed, recompute  Springs From Data Inputs";

    type::vector<Spring>& _springs = *this->springs.beginEdit();
    _springs.clear();



    const SReal& _ks = this->ks.getValue();
    const SReal& _kd = this->kd.getValue();
    for (sofa::Index i = 0; i<indices1.size(); ++i)
        _springs.push_back(Spring(indices1[i], indices2[i], _ks, _kd, lengths[i]));

    this->springs.endEdit();
}



} // namespace sofa::component::solidmechanics::spring
