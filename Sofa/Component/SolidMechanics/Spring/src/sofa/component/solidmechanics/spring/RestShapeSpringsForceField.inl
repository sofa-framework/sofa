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
    : d_points(initData(&d_points, "points", "points controlled by the rest shape springs"))
    , d_stiffness(initData(&d_stiffness, "stiffness", "stiffness values between the actual position and the rest shape position"))
    , d_angularStiffness(initData(&d_angularStiffness, "angularStiffness", "angularStiffness assigned when controlling the rotation of the points"))
    , d_pivotPoints(initData(&d_pivotPoints, "pivot_points", "global pivot points used when translations instead of the rigid mass centers"))
    , d_external_points(initData(&d_external_points, "external_points", "points from the external Mechanical State that define the rest shape springs"))
    , d_recompute_indices(initData(&d_recompute_indices, true, "recompute_indices", "Recompute indices (should be false for BBOX)"))
    , d_drawSpring(initData(&d_drawSpring,false,"drawSpring","draw Spring"))
    , d_springColor(initData(&d_springColor, sofa::type::RGBAColor::green(), "springColor","spring color. (default=[0.0,1.0,0.0,1.0])"))
    , d_activeDirections(initData(&d_activeDirections, s_defaultActiveDirections,
        "activeDirections", std::string("Directions in which the spring is active (default=[" + sofa::helper::join(s_defaultActiveDirections, ',') + "])").c_str()))
    , l_restMState(initLink("external_rest_shape", "rest_shape can be defined by the position of an external Mechanical State"))
    , l_topology(initLink("topology", "Link to be set to the topology container in the component graph"))
{
    this->addUpdateCallback("updateIndices", {&d_points}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        this->recomputeIndices();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription *arg)
{
    const char* attr = arg->getAttribute("external_rest_shape") ;
    if( attr != nullptr && attr[0] != '@')
    {
        msg_error() << "RestShapeSpringsForceField have changed since 17.06. The parameter 'external_rest_shape' is now a Link. To fix your scene you need to add and '@' in front of the provided path. See PR#315" ;
    }

    Inherit::parse(arg) ;
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::bwdInit()
{
    ForceField<DataTypes>::init();

    if (d_stiffness.getValue().empty())
    {
        msg_info() << "No stiffness is defined, assuming equal stiffness on each node, k = 100.0 ";
        d_stiffness.setValue({static_cast<Real>(100)});
    }

    if (l_restMState.get() == nullptr)
    {
        useRestMState = false;
        msg_info() << "no external rest shape used";

        if(!l_restMState.empty())
        {
            msg_warning() << "external_rest_shape in node " << this->getContext()->getName() << " not found";
        }
    }
    else
    {
        msg_info() << "external rest shape used";
        useRestMState = true;
    }

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        d_points.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "Cannot find the topology: topological changes will not be supported";
    }

    recomputeIndices();
    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
        return;

    const BaseMechanicalState* state = this->getContext()->getMechanicalState();
    if(!state)
    {
        msg_warning() << "MechanicalState of the current context returns null pointer";
    }
    else
    {
        assert(state);
        matS.resize(state->getMatrixSize(),state->getMatrixSize());
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
            msg_info() << "'stiffness' is larger than 'angularStiffness', add the default value (100.0) to the missing entries.";

            for(size_t i = as.size();i<s.size();i++)
            {
                as.push_back(100.0);
            }
        }else if (as.size() > s.size())
        {
            msg_info() << "'stiffness' is smaller than 'angularStiffness', clamp the extra values in angularStiffness.";
            as.resize(s.size());
        }
    }

    lastUpdatedStep = -1.0;

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::reinit()
{
    if (!checkOutOfBoundsIndices())
    {
        m_indices.clear();
    }
    else
    {
        msg_info() << "Indices successfully checked";
    }

    if (d_stiffness.getValue().empty())
    {
        msg_info() << "No stiffness is defined, assuming equal stiffness on each node, k = 100.0 " ;

        VecReal stiffs;
        stiffs.push_back(100.0);
        d_stiffness.setValue(stiffs);
    }
    else
    {
        const VecReal &k = d_stiffness.getValue();
        if ( k.size() != m_indices.size() )
        {
            msg_warning() << "Size of stiffness vector is not correct (" << k.size() << "), should be either 1 or " << m_indices.size() << msgendl
                          << "First value of stiffness will be used";
        }
    }

}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::recomputeIndices()
{
    m_indices.clear();
    m_ext_indices.clear();

    for (const sofa::Index i : d_points.getValue())
    {
        m_indices.push_back(i);
    }

    for (const sofa::Index i : d_external_points.getValue())
    {
        m_ext_indices.push_back(i);
    }

    if (m_indices.empty())
    {
        // no point are defined, default case: points = all points
        msg_info() << "No point are defined. Change to default case: points = all points";
        for (sofa::Index i = 0; i < this->mstate->getSize(); i++)
        {
            m_indices.push_back(i);
        }
    }

    if (m_ext_indices.empty())
    {
        if (useRestMState)
        {
            if (const DataVecCoord* extPosition = getExtPosition())
            {
                const auto& extPositionValue = extPosition->getValue();
                for (sofa::Index i = 0; i < extPositionValue.size(); i++)
                {
                    m_ext_indices.push_back(i);
                }
            }
            else
            {
                this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            }
        }
        else
        {
            for (const sofa::Index i : m_indices)
            {
                m_ext_indices.push_back(i);
            }
        }
    }

    if (!checkOutOfBoundsIndices())
    {
        msg_error() << "The dimension of the source and the targeted points are different ";
        m_indices.clear();
    }
    else
    {
        msg_info() << "Indices successfully checked";
    }
}

template<class DataTypes>
bool RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices()
{
    if (!checkOutOfBoundsIndices(m_indices, this->mstate->getSize()))
    {
        msg_error() << "Out of Bounds d_indices detected. ForceField is not activated.";
        return false;
    }
    if (const DataVecCoord* extPosition = getExtPosition())
    {
        if (!checkOutOfBoundsIndices(m_ext_indices, sofa::Size(extPosition->getValue().size())))
        {
            msg_error() << "Out of Bounds m_ext_indices detected. ForceField is not activated.";
            return false;
        }
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    if (m_indices.size() != m_ext_indices.size())
    {
        msg_error() << "Dimensions of the source and the targeted points are different. ForceField is not activated.";
        return false;
    }
    return true;
}

template<class DataTypes>
bool RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices(const VecIndex &indices, const sofa::Size dimension)
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
const typename RestShapeSpringsForceField<DataTypes>::DataVecCoord* RestShapeSpringsForceField<DataTypes>::getExtPosition() const
{
    if(useRestMState)
    {
        if (l_restMState)
        {
            return l_restMState->read(core::vec_id::write_access::position);
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
void RestShapeSpringsForceField<DataTypes>::addForce(const MechanicalParams*  mparams , DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv&  v )
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    WriteAccessor< DataVecDeriv > f1 = f;
    ReadAccessor< DataVecCoord > p1 = x;

    const DataVecCoord* extPosition = getExtPosition();
    if (!extPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    ReadAccessor< DataVecCoord > p0 = *extPosition;

    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();

    f1.resize(p1.size());

    if (d_recompute_indices.getValue())
    {
        recomputeIndices();
    }

    for (sofa::Index i = 0; i < m_indices.size(); i++)
    {
        const sofa::Index index = m_indices[i];
        sofa::Index ext_index = m_indices[i];
        if (useRestMState)
            ext_index = m_ext_indices[i];

        const auto stiffness = k[static_cast<std::size_t>(i < k.size()) * i];

        const auto activeDirections = d_activeDirections.getValue();

        // rigid case
        if constexpr (sofa::type::isRigidType<DataTypes>)
        {
            // translation
            if (i >= m_pivots.size())
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
            }
            else
            {
                CPos localPivot = p0[ext_index].getOrientation().inverseRotate(m_pivots[i] - p0[ext_index].getCenter());
                CPos rotatedPivot = p1[index].getOrientation().rotate(localPivot);
                CPos pivot2 = p1[index].getCenter() + rotatedPivot;
                CPos dx = pivot2 - m_pivots[i];
                getVCenter(f1[index]) -= dx * stiffness;
            }

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
            for (sofa::Size entryId = spatial_dimensions; entryId < Deriv::total_size; ++entryId)
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
void RestShapeSpringsForceField<DataTypes>::addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    WriteAccessor< DataVecDeriv > df1 = df;
    ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto activeDirections = d_activeDirections.getValue();

    for (unsigned int i = 0; i < m_indices.size(); i++)
    {
        const sofa::Index curIndex = m_indices[i];
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
            for (sofa::Size entryId = spatial_dimensions; entryId < Deriv::total_size; ++entryId)
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
            auto currentSpringDx = dx1[m_indices[i]];
            for (sofa::Size entryId = 0; entryId < spatial_dimensions; ++entryId)
            {
                if (!activeDirections[entryId])
                    currentSpringDx[entryId] = 0;
            }
            df1[m_indices[i]] -= currentSpringDx * stiffness * kFactor;
        }
    }


}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::draw(const VisualParams *vparams)
{
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

    const VecIndex& indices = m_indices;
    const VecIndex& ext_indices = (useRestMState ? m_ext_indices : m_indices);

    std::vector<Vec3> vertices;

    for (sofa::Index i=0; i<indices.size(); i++)
    {
        const sofa::Index index = indices[i];
        const sofa::Index ext_index = ext_indices[i];

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

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addKToMatrix(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix )
{
    const MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    BaseMatrix* mat = mref.matrix;
    const unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto activeDirections = d_activeDirections.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    sofa::Index curIndex = 0;

    for (sofa::Index index = 0; index < m_indices.size(); index++)
    {
        curIndex = m_indices[index];

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
void RestShapeSpringsForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    const VecReal& k = d_stiffness.getValue();
    const VecReal& k_a = d_angularStiffness.getValue();
    const auto activeDirections = d_activeDirections.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto index : m_indices)
    {
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
void RestShapeSpringsForceField<DataTypes>::buildDampingMatrix(
    core::behavior::DampingMatrix* matrix)
{
    SOFA_UNUSED(matrix);
}
} // namespace sofa::component::solidmechanics::spring
