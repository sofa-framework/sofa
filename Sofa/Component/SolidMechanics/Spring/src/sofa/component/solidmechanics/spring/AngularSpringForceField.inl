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
#include <sofa/component/solidmechanics/spring/AngularSpringForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <cassert>


namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
AngularSpringForceField<DataTypes>::AngularSpringForceField()
    : d_indices(initData(&d_indices, "indices", "index of nodes controlled by the angular springs"))
    , d_angularStiffness(initData(&d_angularStiffness, "angularStiffness", "angular stiffness for the controlled nodes"))
    , d_angularLimit(initData(&d_angularLimit, "limit", "angular limit (max; min) values where the force applies"))
    , d_drawSpring(initData(&d_drawSpring, false, "drawSpring", "draw Spring"))
    , d_springColor(initData(&d_springColor, type::RGBAColor::green(), "springColor", "spring color"))
{
    indices.setOriginalData(&d_indices);
    angularStiffness.setOriginalData(&d_angularStiffness);
    angularLimit.setOriginalData(&d_angularLimit);
    drawSpring.setOriginalData(&d_drawSpring);
    springColor.setOriginalData(&d_springColor);

}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::bwdInit()
{
    core::behavior::ForceField<DataTypes>::init();

    if (d_angularStiffness.getValue().empty())
    {
		msg_info("AngularSpringForceField") << "No angular stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << "\n";

        VecReal stiffs;
        stiffs.push_back(100.0);
        d_angularStiffness.setValue(stiffs);
    }
    this->k = d_angularStiffness.getValue();

    mState = dynamic_cast<core::behavior::MechanicalState<DataTypes> *> (this->getContext()->getMechanicalState());
    if (!mState) {
		msg_error("AngularSpringForceField") << "MechanicalStateFilter has no binding MechanicalState" << "\n";
    }
    matS.resize(mState->getMatrixSize(), mState->getMatrixSize());
}

template<class DataTypes>
void AngularSpringForceField<DataTypes>::reinit()
{
    if (d_angularStiffness.getValue().empty())
    {
		msg_info("AngularSpringForceField") << "nN angular stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << "\n";

        VecReal stiffs;
        stiffs.push_back(100.0);
        d_angularStiffness.setValue(stiffs);
    }
    this->k = d_angularStiffness.getValue();
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    if(!mState) {
		msg_info("AngularSpringForceField") << "No Mechanical State found, no force will be computed..." << "\n";
        return;
    }

    sofa::helper::WriteAccessor< DataVecDeriv > f1 = f;
    sofa::helper::ReadAccessor< DataVecCoord > p1 = x;
    f1.resize(p1.size());
    for (sofa::Index i = 1; i < d_indices.getValue().size(); i++)
    {
        const sofa::Index index = d_indices.getValue()[i];
        type::Quat<SReal> dq = p1[index].getOrientation() * p1[index-1].getOrientation().inverse();
        type::Vec3d axis;
        double angle = 0.0;
        Real stiffness;
        dq.normalize();

        Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

		// to avoid numerical instabilities of acos for theta < 5
		if(dq[3]>0.999999) // theta < 5 -> q[3] = cos(theta/2) > 0.999
        {
            sin_half_theta = sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2]);
            angle = (Real)(2.0 * asin(sin_half_theta));
        }
        else
        {
            Real half_theta = acos(dq[3]);
            sin_half_theta = sin(half_theta);
            angle = 2*half_theta;
        }

        assert(sin_half_theta>=0);
        if (sin_half_theta < std::numeric_limits<Real>::epsilon())
            axis = type::Vec<3,Real>(0.0, 1.0, 0.0);
        else
            axis = type::Vec<3,Real>(dq[0], dq[1], dq[2])/sin_half_theta;

		if (i < this->k.size())
            stiffness = this->k[i] = d_angularStiffness.getValue()[i];
         else
            stiffness = this->k[0] = d_angularStiffness.getValue()[0];

        getVOrientation(f1[index]) -= axis * angle * stiffness;
    }
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< DataVecDeriv > df1 = df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx1 = dx;

    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    for (sofa::Index i=0; i < d_indices.getValue().size(); i++)
        getVOrientation(df1[d_indices.getValue()[i]]) -= getVOrientation(dx1[d_indices.getValue()[i]]) * (i < this->k.size() ? this->k[i] : this->k[0]) * kFactor ;
}


template<class DataTypes>
void AngularSpringForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const int N = 6;
    const sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::linearalgebra::BaseMatrix* mat = mref.matrix;
    const unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    sofa::Index curIndex = 0;
    for (sofa::Index index = 0; index < d_indices.getValue().size(); index++)
    {
        curIndex = d_indices.getValue()[index];
        for(int i = 3; i < 6; i++)
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * (index < this->k.size() ? this->k[index] : this->k[0]));
    }
}

template <class DataTypes>
void AngularSpringForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);
    assert(!k.empty());

    const auto& indicesValue = d_indices.getValue();
    const auto addValueToMatrix = [&dfdx](const sofa::Index nodeIndex, Real v)
    {
        for(sofa::Size j = Deriv::spatial_dimensions; j < Deriv::total_size; ++j)
        {
            const sofa::Size row = Deriv::total_size * nodeIndex + j;
            dfdx(row, row) += v;
        }
    };

    //separate the loop in 2 in case k.size() != indicesValue.size()
    const auto minSize = std::min(indicesValue.size(), this->k.size());
    for (std::size_t i = 0; i < minSize; ++i)
    {
        addValueToMatrix(indicesValue[i], -this->k[i]);
    }

    for (std::size_t i = minSize; i < indicesValue.size(); ++i)
    {
        addValueToMatrix(indicesValue[i], -this->k[0]);
    }
}

template <class DataTypes>
void AngularSpringForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void AngularSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !d_drawSpring.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->setLightingEnabled(false);

    sofa::helper::ReadAccessor< DataVecCoord > p = this->mstate->read(core::vec_id::write_access::position);
    sofa::type::vector< type::Vec3 > vertices;

    for (sofa::Index i=0; i < d_indices.getValue().size(); i++)
    {
        const sofa::Index index = d_indices.getValue()[i];
        vertices.push_back(p[index].getCenter());
    }
    vparams->drawTool()->drawLines(vertices, 5, d_springColor.getValue());

}

} // namespace sofa::component::solidmechanics::spring
