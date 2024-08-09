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

template<class DataTypes>
FixedWeakConstraint<DataTypes>::FixedWeakConstraint()
    : d_fixAll(initData(&d_fixAll, false,"fixAll", "stiffness value between the actual position and the rest shape position"))
{}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::init()
{
    Inherit::init();
    if (this->l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        this->l_topology.set(this->getContext()->getMeshTopologyLink());
        if(! this->l_topology.get())
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = this->l_topology.get())
    {
        msg_info() << "Topology path used: '" << this->l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        this->d_indices.createTopologyHandler(_topology);
    }


    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class DataTypes>
bool FixedWeakConstraint<DataTypes>::checkOutOfBoundsIndices()
{
    for(auto idx : this->d_indices.getValue())
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
void FixedWeakConstraint<DataTypes>::addForce(const core::MechanicalParams*  mparams , DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv&  v )
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    helper::WriteAccessor< DataVecDeriv > f1 = f;
    helper::ReadAccessor< DataVecCoord > p1 = x;

    const DataVecCoord * restPosition = this->mstate->read(sofa::core::VecId::resetPosition());
    if (!restPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    helper::ReadAccessor< DataVecCoord > p0 = *restPosition;

    const auto& stiffness = this->d_stiffness.getValue();
    const auto& angularStiffness = this->d_angularStiffness.getValue();

    f1.resize(p1.size());


    for (sofa::Index i = 0; i < this->d_indices.getValue().size(); i++)
    {
        const sofa::Index index = this->d_indices.getValue()[i];

        // rigid case
        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            // translation
            CPos dx = p1[index].getCenter() - p0[index].getCenter();
            getVCenter(f1[index]) -= dx * stiffness[i];

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

            getVOrientation(f1[index]) -= dir * angle * angularStiffness[i];
        }
        else // non-rigid implementation
        {
            Deriv dx = p1[index] - p0[index];
            f1[index] -= dx * stiffness[i];
        }
    }
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    helper::WriteAccessor< DataVecDeriv > df1 = df;
    helper::ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const auto& stiffness = this->d_stiffness.getValue();
    const auto& angularStiffness = this->d_angularStiffness.getValue();

    for (unsigned int i = 0; i < this->d_indices.getValue().size(); i++)
    {
        const sofa::Index curIndex = this->d_indices.getValue()[i];

        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            getVCenter(df1[curIndex]) -= getVCenter(dx1[curIndex]) * stiffness[i] * kFactor;
            getVOrientation(df1[curIndex]) -= getVOrientation(dx1[curIndex]) * angularStiffness[i] * kFactor;
        }
        else
        {
            df1[this->d_indices.getValue()[i]] -= dx1[this->d_indices.getValue()[i]] * stiffness[i] * kFactor;
        }
    }


}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::draw(const  core::visual::VisualParams *vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !this->d_drawSpring.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->setLightingEnabled(false);

    const DataVecCoord * restPosition = this->mstate->read(sofa::core::VecId::resetPosition());
    if (!restPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }


    helper::ReadAccessor< DataVecCoord > p0 = *restPosition;
    helper::ReadAccessor< DataVecCoord > p  = this->mstate->read(core::VecCoordId::position());

    const VecIndex& indices = this->d_indices.getValue();

    std::vector<type::Vec3> vertices;

    for (sofa::Index i=0; i<indices.size(); i++)
    {
        const sofa::Index index = indices[i];

        type::Vec3 v0(0.0, 0.0, 0.0);
        type::Vec3 v1(0.0, 0.0, 0.0);
        for(sofa::Index j=0 ; j< std::min(DataTypes::spatial_dimensions, static_cast<sofa::Size>(3)) ; j++)
        {
            v0[j] = (DataTypes::getCPos(p[index]))[j];
            v1[j] = (DataTypes::getCPos(p0[index]))[j];
        }

        vertices.push_back(v0);
        vertices.push_back(v1);
    }

    vparams->drawTool()->drawLines(vertices,5, this->d_springColor.getValue());
}

template<class DataTypes>
void FixedWeakConstraint<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const core::behavior::MultiMatrixAccessor* matrix )
{
    const core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    linearalgebra::BaseMatrix* mat = mref.matrix;
    const unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const auto& k = this->d_stiffness.getValue();
    const auto& k_a = this->d_angularStiffness.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    sofa::Index curIndex = 0;

    for (sofa::Index index = 0; index < this->d_indices.getValue().size(); index++)
    {
        curIndex = this->d_indices.getValue()[index];

        // translation
        const auto vt = -kFact * k[index];
        for (sofa::Size i = 0; i < space_size; i++)
        {
            mat->add(offset + total_size * curIndex + i, offset + total_size * curIndex + i, vt);
        }

        // rotation (if applicable)
        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            const auto vr = -kFact * k_a[index];
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
    const auto& vt = this->d_stiffness.getValue();
    const auto& vr = this->d_angularStiffness.getValue();

    constexpr sofa::Size space_size = Deriv::spatial_dimensions; // == total_size if DataTypes = VecTypes
    constexpr sofa::Size total_size = Deriv::total_size;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);


    const VecIndex& indices = this->d_indices.getValue();
    for (sofa::Index i=0; i<indices.size(); i++)
    {
        const sofa::Index index = indices[i];

        // translation
        for(sofa::Index i = 0; i < space_size; i++)
        {
            dfdx(total_size * index + i, total_size * index + i) += -vt[i];
        }

        // rotation (if applicable)
        if constexpr (sofa::type::isRigidType<DataTypes>())
        {
            for (sofa::Size i = space_size; i < total_size; ++i)
            {
                dfdx(total_size * index + i, total_size * index + i) += -vr[i];
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
