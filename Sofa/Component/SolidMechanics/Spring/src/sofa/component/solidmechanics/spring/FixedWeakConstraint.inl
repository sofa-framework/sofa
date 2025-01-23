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
#include <sofa/component/solidmechanics/spring/FixedWeakConstraint.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/type/isRigidType.h>

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
FixedWeakConstraint<DataTypes>::FixedWeakConstraint()
    : d_indices(initData(&d_indices, "indices", "points controlled by the rest shape springs"))
    , d_fixAll(initData(&d_fixAll, false, "fixAll", "Force to fix all points"))
    , d_stiffness(initData(&d_stiffness, "stiffness", "stiffness values between the actual position and the rest shape position"))
    , d_angularStiffness(initData(&d_angularStiffness, "angularStiffness", "angularStiffness assigned when controlling the rotation of the points"))
    , d_drawSpring(initData(&d_drawSpring,false,"drawSpring","draw Spring"))
    , d_springColor(initData(&d_springColor, sofa::type::RGBAColor::green(), "springColor","spring color. (default=[0.0,1.0,0.0,1.0])"))
    , l_topology(initLink("topology", "Link to be set to the topology container in the component graph"))
{
    this->addUpdateCallback("updateIndices", {&d_indices}, [this](const core::DataTracker& t)
                            {
                                SOFA_UNUSED(t);
                                if (!checkOutOfBoundsIndices())
                                {
                                    msg_error(this) << "Some input indices are out of bound";
                                    return sofa::core::objectmodel::ComponentState::Invalid;
                                }
                                else
                                {
                                    return sofa::core::objectmodel::ComponentState::Valid;
                                }
                            }, {});
}


template<class DataTypes>
const bool  FixedWeakConstraint<DataTypes>::checkState()
{
    if (d_stiffness.getValue().empty())
    {
        msg_info(this) << "No stiffness is defined, assuming equal stiffness on each node, k = 100.0 ";
        d_stiffness.setValue({static_cast<Real>(100)});
    }

    if (l_topology.empty())
    {
        msg_info(this) << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info(this) << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info(this) << "Cannot find the topology: topological changes will not be supported";
    }



    /// Compile time condition to check if we are working with a Rigid3Types or a type that does not
    /// need the Angular Stiffness parameters.
    //if constexpr (isRigid())
    if constexpr (sofa::type::isRigidType<DataTypes>)
    {
        sofa::helper::ReadAccessor<Data<VecReal>> s = d_stiffness;
        sofa::helper::WriteOnlyAccessor<Data<VecReal>> as = d_angularStiffness;

        if (as.size() < s.size())
        {
            msg_info(this) << "'stiffness' is larger than 'angularStiffness', add the default value (100.0) to the missing entries.";

            for(size_t i = as.size();i<s.size();i++)
            {
                as.push_back(100.0);
            }
        }else if (as.size() > s.size())
        {
            msg_info(this) << "'stiffness' is smaller than 'angularStiffness', clamp the extra values in angularStiffness.";
            as.resize(s.size());
        }
    }

    if (!checkOutOfBoundsIndices())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::bwdInit()
{
    ForceField<DataTypes>::init();

    if (checkState())
    {
        this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }
    else
    {
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);

    }
}


template<class DataTypes>
void FixedWeakConstraint<DataTypes>::reinit()
{
    ForceField<DataTypes>::reinit();

    if (checkState())
    {
        this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }
    else
    {
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);

    }

}



template<class DataTypes>
bool FixedWeakConstraint<DataTypes>::checkOutOfBoundsIndices()
{
    if (!checkOutOfBoundsIndices(getIndices(), this->mstate->getSize()))
    {
        return false;
    }
    return true;
}

template<class DataTypes>
bool FixedWeakConstraint<DataTypes>::checkOutOfBoundsIndices(const VecIndex &indices, const sofa::Size dimension)
{
    for (sofa::Index i = 0; i < indices.size(); i++)
    {
        if (indices[i] >= dimension)
        {
            return false;
        }
    }
    return true;
}

template<class DataTypes>
const typename FixedWeakConstraint<DataTypes>::DataVecCoord* FixedWeakConstraint<DataTypes>::getExtPosition() const
{
    if (this->mstate)
    {
        return this->mstate->read(core::vec_id::write_access::restPosition);
    }
    return nullptr;
}

template<class DataTypes>
const typename FixedWeakConstraint<DataTypes>::VecIndex& FixedWeakConstraint<DataTypes>::getIndices() const
{

    return d_indices.getValue();
}

template<class DataTypes>
const typename FixedWeakConstraint<DataTypes>::VecIndex& FixedWeakConstraint<DataTypes>::getExtIndices() const
{
    return d_indices.getValue();
}

template<class DataTypes>
const type::fixed_array<bool, FixedWeakConstraint<DataTypes>::coord_total_size>& FixedWeakConstraint<DataTypes>::getActiveDirections() const
{
    return FixedWeakConstraint<DataTypes>::s_defaultActiveDirections;
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addForce(const MechanicalParams*  mparams , DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv&  v )
{
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return;

    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    WriteAccessor< DataVecDeriv > f1 = f;
    ReadAccessor< DataVecCoord > p1 = x;

    const DataVecCoord* extPosition = getExtPosition();
    const auto & indices = getIndices();
    const auto & extIndices = getExtIndices();
    if (!extPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    ReadAccessor< DataVecCoord > p0 = *extPosition;

    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();


    f1.resize(p1.size());

    const bool fixedAll = d_fixAll.getValue();
    const unsigned maxIt = fixedAll ? this->mstate->getSize() : indices.size();

    for (sofa::Index i = 0; i < maxIt; i++)
    {
        sofa::Index ext_index = i;
        sofa::Index index = i;

        if (!fixedAll)
        {
            index = indices[i];
            ext_index = extIndices[i];
        }

        const auto stiffness = k[static_cast<std::size_t>(i < k.size()) * i];

        const auto & activeDirections = getActiveDirections();

        // rigid case
        if constexpr (sofa::type::isRigidType<DataTypes>)
        {

            CPos dx = p1[index].getCenter() - p0[ext_index].getCenter();
            // We filter the difference dx by setting to 0 the entries corresponding
            // to 0 values in d_activeDirections
            for (sofa::Size entryId = 0; entryId < spatial_dimensions; ++entryId)
            {
                if (!activeDirections[entryId])
                    dx[entryId] = 0;
            }
            getVCenter(f1[index]) -= dx * stiffness;



            // rotation
            type::Quat<Real> dq = p1[index].getOrientation() * p0[ext_index].getOrientation().inverse();
            dq.normalize();

            type::Vec<3, Real> dir{type::NOINIT};
            Real angle = 0.;

            if (dq[3] < 0.)
            {
                dq = dq * -1.0;
            }

            if (dq[3] < 1.0)
                dq.quatToAxis(dir, angle);

            // We change the direction of the axis of rotation based on
            // the 0 values in d_activeDirections. This is equivalent
            // to senting to 0 the rotation axis components along x, y
            // and/or z, depending on the rotations we want to take into
            // account.
            for (sofa::Size entryId = spatial_dimensions; entryId < coord_total_size; ++entryId)
            {
                if (!activeDirections[entryId])
                    dir[entryId-spatial_dimensions] = 0;
            }

            const auto angularStiffness = k_a[static_cast<std::size_t>(i < k_a.size()) * i];
            getVOrientation(f1[index]) -= dir * angle * angularStiffness;
        }
        else // non-rigid implementation
        {
            Deriv dx = p1[index] - p0[ext_index];
            // We filter the difference dx by setting to 0 the entries corresponding
            // to 0 values in d_activeDirections
            for (sofa::Size entryId = 0; entryId < spatial_dimensions; ++entryId)
            {
                if (!activeDirections[entryId])
                    dx[entryId] = 0;
            }
            f1[index] -= dx * stiffness;
        }
    }
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return;

    WriteAccessor< DataVecDeriv > df1 = df;
    ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto & activeDirections = getActiveDirections();

    const auto & indices = getIndices();

    const bool fixedAll = d_fixAll.getValue();
    const unsigned maxIt = fixedAll ? this->mstate->getSize() : indices.size();

    for (sofa::Index i = 0; i < maxIt; i++)
    {
        sofa::Index curIndex = i;

        if (!fixedAll)
        {
            curIndex = indices[i];
        }

        const auto stiffness = k[static_cast<std::size_t>(i < k.size()) * i];

        if constexpr (sofa::type::isRigidType<DataTypes>)
        {
            const auto angularStiffness = k_a[static_cast<std::size_t>(i < k_a.size()) * i];

            // We filter the difference in translation by setting to 0 the entries corresponding
            // to 0 values in d_activeDirections
            auto currentSpringDx = getVCenter(dx1[curIndex]);
            for (sofa::Size entryId = 0; entryId < spatial_dimensions; ++entryId)
            {
                if (!activeDirections[entryId])
                    currentSpringDx[entryId] = 0;
            }
            getVCenter(df1[curIndex]) -= currentSpringDx * stiffness * kFactor;

            auto currentSpringRotationalDx = getVOrientation(dx1[curIndex]);
            // We change the direction of the axis of rotation based on
            // the 0 values in d_activeDirections. This is equivalent
            // to senting to 0 the rotation axis components along x, y
            // and/or z, depending on the rotations we want to take into
            // account.
            for (sofa::Size entryId = spatial_dimensions; entryId < coord_total_size; ++entryId)
            {
                if (!activeDirections[entryId])
                    currentSpringRotationalDx[entryId-spatial_dimensions] = 0;
            }
            getVOrientation(df1[curIndex]) -= currentSpringRotationalDx * angularStiffness * kFactor;
        }
        else
        {
            // We filter the difference in translation by setting to 0 the entries corresponding
            // to 0 values in d_activeDirections
            auto currentSpringDx = dx1[curIndex];
            for (sofa::Size entryId = 0; entryId < spatial_dimensions; ++entryId)
            {
                if (!activeDirections[entryId])
                    currentSpringDx[entryId] = 0;
            }
            df1[curIndex] -= currentSpringDx * stiffness * kFactor;
        }
    }


}

template<class DataTypes>
SReal FixedWeakConstraint<DataTypes>::getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);

    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addKToMatrix(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix )
{
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return;

    const MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    BaseMatrix* mat = mref.matrix;
    const unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto & activeDirections = getActiveDirections();
    const auto & indices = getIndices();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    const bool fixedAll = d_fixAll.getValue();
    const unsigned maxIt = fixedAll ? this->mstate->getSize() : indices.size();

    for (sofa::Index index = 0; index < maxIt; index++)
    {
        sofa::Index curIndex = index;

        if (!fixedAll)
        {
            curIndex = indices[index];
        }

        // translation
        const auto vt = -kFact * k[(index < k.size()) * index];
        for (sofa::Size i = 0; i < space_size; i++)
        {
            // Contribution to the stiffness matrix are only taken into
            // account for 1 values in d_activeDirections
            if (activeDirections[i])
                mat->add(offset + total_size * curIndex + i, offset + total_size * curIndex + i, vt);
        }

        // rotation (if applicable)
        if constexpr (sofa::type::isRigidType<DataTypes>)
        {
            const auto vr = -kFact * k_a[(index < k_a.size()) * index];
            for (sofa::Size i = space_size; i < total_size; i++)
            {
                // Contribution to the stiffness matrix are only taken into
                // account for 1 values in d_activeDirections
                if (activeDirections[i])
                    mat->add(offset + total_size * curIndex + i, offset + total_size * curIndex + i, vr);
            }
        }
    }
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return;

    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto & activeDirections = getActiveDirections();
    const auto & indices = getIndices();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                    .withRespectToPositionsIn(this->mstate);
    const bool fixedAll = d_fixAll.getValue();
    const unsigned maxIt = fixedAll ? this->mstate->getSize() : indices.size();

    for (sofa::Index i = 0; i < maxIt; i++)
    {
        sofa::Index index = i;

        if (!fixedAll)
        {
            index = indices[i];
        }
        // translation
        const auto vt = -k[(index < k.size()) * index];
        for(sofa::Index i = 0; i < space_size; i++)
        {
            dfdx(total_size * index + i, total_size * index + i) += vt;
        }

        // rotation (if applicable)
        if constexpr (sofa::type::isRigidType<DataTypes>)
        {
            const auto vr = -k_a[(index < k_a.size()) * index];
            for (sofa::Size i = space_size; i < total_size; ++i)
            {
                // Contribution to the stiffness matrix are only taken into
                // account for 1 values in d_activeDirections
                if (activeDirections[i])
                {
                    dfdx(total_size * index + i, total_size * index + i) += vr;
                }
            }
        }
    }
}

template <class DataTypes>
void FixedWeakConstraint<DataTypes>::buildDampingMatrix(
    core::behavior::DampingMatrix* matrix)
{
    SOFA_UNUSED(matrix);
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::draw(const VisualParams *vparams)
{
    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Valid)
        return;

    if (!vparams->displayFlags().getShowForceFields() || !d_drawSpring.getValue())
        return;  /// \todo put this in the parent class

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->setLightingEnabled(false);

    const DataVecCoord* extPosition = getExtPosition();
    if (!extPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    ReadAccessor< DataVecCoord > p0 = *extPosition;
    ReadAccessor< DataVecCoord > p  = this->mstate->read(sofa::core::vec_id::write_access::position);
    std::vector<Vec3> vertices;


    const auto & indices = getIndices();
    const auto & extIndices = getExtIndices();
    const bool fixedAll = d_fixAll.getValue();
    const unsigned maxIt = fixedAll ? this->mstate->getSize() : indices.size();

    for (sofa::Index i = 0; i < maxIt; i++)
    {
        sofa::Index ext_index = i;
        sofa::Index index = i;

        if (!fixedAll)
        {
            index = indices[i];
            ext_index = extIndices[i];
        }

        Vec3 v0(0.0, 0.0, 0.0);
        Vec3 v1(0.0, 0.0, 0.0);
        for(sofa::Index j=0 ; j< std::min(DataTypes::spatial_dimensions, static_cast<sofa::Size>(3)) ; j++)
        {
            v0[j] = (DataTypes::getCPos(p[index]))[j];
            v1[j] = (DataTypes::getCPos(p0[ext_index]))[j];
        }

        vertices.push_back(v0);
        vertices.push_back(v1);
    }

    //todo(dmarchal) because of https://github.com/sofa-framework/sofa/issues/64
    vparams->drawTool()->drawLines(vertices,5, d_springColor.getValue());

}
} // namespace sofa::component::solidmechanics::spring
