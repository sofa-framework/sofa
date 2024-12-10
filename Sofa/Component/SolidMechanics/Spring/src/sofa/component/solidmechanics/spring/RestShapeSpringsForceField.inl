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
#include <sofa/component/solidmechanics/spring/RestShapeSpringsForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/type/isRigidType.h>
#include <sofa/component/solidmechanics/spring/FixedWeakConstraint.inl>

#include <string_view>
#include <type_traits>

namespace sofa::component::solidmechanics::spring
{

using helper::WriteAccessor;
using helper::ReadAccessor;
using core::behavior::BaseMechanicalState;
using core::behavior::MultiMatrixAccessor;
using core::behavior::ForceField;
using linearalgebra::BaseMatrix;
using core::VecCoordId;
using core::MechanicalParams;
using type::Vec3;
using type::Vec4f;
using type::vector;
using core::visual::VisualParams;

template<class DataTypes>
RestShapeSpringsForceField<DataTypes>::RestShapeSpringsForceField()
    : d_activeDirections(initData(&d_activeDirections, FixedWeakConstraint<DataTypes>::s_defaultActiveDirections,
        "activeDirections", std::string("Directions in which the spring is active (default=[" + sofa::helper::join(FixedWeakConstraint<DataTypes>::s_defaultActiveDirections, ',') + "])").c_str()))
    , d_externalIndices(initData(&d_externalIndices, "externalIndices","Indices from the external Mechanical State that define the rest shape springs"))
    , l_restMState(initLink("external_rest_shape", "rest_shape can be defined by the position of an external Mechanical State"))
    , l_topology(initLink("topology", "Link to be set to the topology container in the component graph"))
{
    c_fixAllCallback.addInput(&this->d_fixAll);
    c_fixAllCallback.addCallback([this]()
    {

        if (this->getMState()->getSize() != l_restMState->getSize())
        {
            msg_error(this) <<"the fixAll option only works with either one mstate or when the two mstate have the same size.";
            this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        }
        else
        {
            this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
        }
    });
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::bwdInit()
{
    FixedWeakConstraint<DataTypes>::bwdInit();

    if (l_restMState.get() == nullptr)
    {
        m_useRestMState = false;
        msg_info(this) << "no external rest shape used";

        if(!l_restMState.empty())
        {
            msg_warning() << "external_rest_shape in node " << this->getContext()->getName() << " not found";
        }
    }
    else
    {
        msg_info(this) << "external rest shape used";
        m_useRestMState = true;
    }
}


template<class DataTypes>
bool RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices()
{
    if (!FixedWeakConstraint<DataTypes>::checkOutOfBoundsIndices(this->getIndices(), this->mstate->getSize()))
    {
        msg_error() << "Out of Bounds d_indices detected. ForceField is not activated.";
        return false;
    }
    if (const DataVecCoord* extPosition = getExtPosition())
    {
        if (!FixedWeakConstraint<DataTypes>::checkOutOfBoundsIndices(getExtIndices(), sofa::Size(extPosition->getValue().size())))
        {
            msg_error() << "Out of Bounds m_ext_indices detected. ForceField is not activated.";
            return false;
        }
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    if (this->getIndices().size() != getExtIndices().size())
    {
        msg_error() << "Dimensions of the source and the targeted points are different. ForceField is not activated.";
        return false;
    }
    return true;
}

template<class DataTypes>
const typename RestShapeSpringsForceField<DataTypes>::DataVecCoord* RestShapeSpringsForceField<DataTypes>::getExtPosition() const
{
    if(m_useRestMState)
    {
        if (l_restMState)
        {
            return l_restMState->read(core::vec_id::write_access::position);
        }
        else
        {
            msg_error(this)<<"The external rest shape is not set correctly.";
        }
    }
    else
    {
        if (this->mstate)
        {
            return this->mstate->read(core::vec_id::write_access::restPosition);
        }
    }
    return nullptr;
}


template<class DataTypes>
const typename RestShapeSpringsForceField<DataTypes>::VecIndex& RestShapeSpringsForceField<DataTypes>::getExtIndices() const
{
    if (m_useRestMState )
    {
        return d_externalIndices.getValue();
    }
    else
    {
        return this->d_indices.getValue();
    }
}

template<class DataTypes>
const type::fixed_array<bool, RestShapeSpringsForceField<DataTypes>::coord_total_size>& RestShapeSpringsForceField<DataTypes>::getActiveDirections() const
{
    return d_activeDirections.getValue();
}

} // namespace sofa::component::solidmechanics::spring
