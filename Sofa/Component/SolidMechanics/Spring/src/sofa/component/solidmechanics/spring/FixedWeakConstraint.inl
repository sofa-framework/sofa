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
    , d_stiffness(initData(&d_stiffness, 1.0,"stiffness", "stiffness value between the actual position and the rest shape position"))
    , d_angularStiffness(initData(&d_angularStiffness,1.0, "angularStiffness", "angularStiffness assigned when controlling the rotation of the points"))
    , d_drawSpring(initData(&d_drawSpring,false,"drawSpring","draw Spring"))
    , d_springColor(initData(&d_springColor, sofa::type::RGBAColor::green(), "springColor","spring color. (default=[0.0,1.0,0.0,1.0])"))
    , l_topology(initLink("topology", "Link to be set to the topology container in the component graph"))
{
    this->addUpdateCallback("updateInputs", {&d_indices}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        if(checkOutOfBoundsIndices())
        {
            msg_error() << "Input indices out of bound";
            return sofa::core::objectmodel::ComponentState::Invalid;
        }

        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::init()
{
    Inherit::init();
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
        if(! l_topology.get())
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class DataTypes>
bool FixedWeakConstraint<DataTypes>::checkOutOfBoundsIndices()
{
    for(auto idx : d_indices.getValue())
    {
        if(idx >= this->mstate->getSize())
            return true;
    }
    return false;
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
void FixedWeakConstraint<DataTypes>::addForce(const MechanicalParams*  mparams , DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv&  v )
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    WriteAccessor< DataVecDeriv > f1 = f;
    ReadAccessor< DataVecCoord > p1 = x;

    const DataVecCoord * restPosition = this->mstate->read(sofa::core::VecId::resetPosition());
    if (!restPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    ReadAccessor< DataVecCoord > p0 = *restPosition;

    const Real& stiffness = d_stiffness.getValue();
    const Real& angularStiffness = d_angularStiffness.getValue();

    f1.resize(p1.size());


    for (sofa::Index i = 0; i < d_indices.getValue().size(); i++)
    {
        const sofa::Index index = d_indices.getValue()[i];

        // rigid case
        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            // translation
            CPos dx = p1[index].getCenter() - p0[index].getCenter();
            getVCenter(f1[index]) -= dx * stiffness;

            // rotation
            type::Quat<Real> dq = p1[index].getOrientation() * p0[index].getOrientation().inverse();
            dq.normalize();

            type::Vec<3, Real> dir{type::NOINIT};
            Real angle = 0.;

            if (dq[3] < 0.)
            {
                dq = dq * -1.0;
            }

            if (dq[3] < 1.0)
                dq.quatToAxis(dir, angle);

            getVOrientation(f1[index]) -= dir * angle * angularStiffness;
        }
        else // non-rigid implementation
        {
            Deriv dx = p1[index] - p0[index];
            f1[index] -= dx * stiffness;
        }
    }
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    WriteAccessor< DataVecDeriv > df1 = df;
    ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const Real& stiffness = d_stiffness.getValue();
    const Real& angularStiffness = d_angularStiffness.getValue();

    for (unsigned int i = 0; i < d_indices.getValue().size(); i++)
    {
        const sofa::Index curIndex = d_indices.getValue()[i];

        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            getVCenter(df1[curIndex]) -= getVCenter(dx1[curIndex]) * stiffness * kFactor;
            getVOrientation(df1[curIndex]) -= getVOrientation(dx1[curIndex]) * angularStiffness * kFactor;
        }
        else
        {
            df1[d_indices.getValue()[i]] -= dx1[d_indices.getValue()[i]] * stiffness * kFactor;
        }
    }


}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::draw(const VisualParams *vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !d_drawSpring.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->setLightingEnabled(false);

    const DataVecCoord * restPosition = this->mstate->read(sofa::core::VecId::resetPosition());
    if (!restPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }


    ReadAccessor< DataVecCoord > p0 = *restPosition;
    ReadAccessor< DataVecCoord > p  = this->mstate->read(VecCoordId::position());

    const VecIndex& indices = d_indices.getValue();

    std::vector<Vec3> vertices;

    for (sofa::Index i=0; i<indices.size(); i++)
    {
        const sofa::Index index = indices[i];

        Vec3 v0(0.0, 0.0, 0.0);
        Vec3 v1(0.0, 0.0, 0.0);
        for(sofa::Index j=0 ; j< std::min(DataTypes::spatial_dimensions, static_cast<sofa::Size>(3)) ; j++)
        {
            v0[j] = (DataTypes::getCPos(p[index]))[j];
            v1[j] = (DataTypes::getCPos(p0[index]))[j];
        }

        vertices.push_back(v0);
        vertices.push_back(v1);
    }

    vparams->drawTool()->drawLines(vertices,5, d_springColor.getValue());
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addKToMatrix(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix )
{
    const MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    BaseMatrix* mat = mref.matrix;
    const unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const Real& k = d_stiffness.getValue();
    const Real& k_a = d_angularStiffness.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    sofa::Index curIndex = 0;

    for (sofa::Index index = 0; index < d_indices.getValue().size(); index++)
    {
        curIndex = d_indices.getValue()[index];

        // translation
        const auto vt = -kFact * k;
        for (sofa::Size i = 0; i < space_size; i++)
        {
            mat->add(offset + total_size * curIndex + i, offset + total_size * curIndex + i, vt);
        }

        // rotation (if applicable)
        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            const auto vr = -kFact * k_a;
            for (sofa::Size i = space_size; i < total_size; i++)
            {
                mat->add(offset + total_size * curIndex + i, offset + total_size * curIndex + i, vr);
            }
        }
    }
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    const Real& vt = -d_stiffness.getValue();
    const Real& vr = -d_angularStiffness.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto index : d_indices.getValue())
    {
        // translation
        for(sofa::Index i = 0; i < space_size; i++)
        {
            dfdx(total_size * index + i, total_size * index + i) += vt;
        }

        // rotation (if applicable)
        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            for (sofa::Size i = space_size; i < total_size; ++i)
            {
                dfdx(total_size * index + i, total_size * index + i) += vr;
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
} // namespace sofa::component::solidmechanics::spring
