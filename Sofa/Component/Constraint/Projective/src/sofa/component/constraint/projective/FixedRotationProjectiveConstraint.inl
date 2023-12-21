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

#include <sofa/component/constraint/projective/FixedRotationProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <utility>


namespace sofa::component::constraint::projective
{


template <class DataTypes>
FixedRotationProjectiveConstraint<DataTypes>::FixedRotationProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr),
      FixedXRotation( initData( &FixedXRotation, false, "FixedXRotation", "Prevent Rotation around X axis")),
      FixedYRotation( initData( &FixedYRotation, false, "FixedYRotation", "Prevent Rotation around Y axis")),
      FixedZRotation( initData( &FixedZRotation, false, "FixedZRotation", "Prevent Rotation around Z axis"))
{
}


template <class DataTypes>
FixedRotationProjectiveConstraint<DataTypes>::~FixedRotationProjectiveConstraint()
{
}


template <class DataTypes>
void FixedRotationProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    // Retrieves mechanical state
    VecCoord x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    // Stores initial orientation for each vertex
    previousOrientation.resize(x.size());
    for (unsigned int i=0; i<previousOrientation.size(); i++)
    {
        previousOrientation[i] = x[i].getOrientation();
    }
}

template <class DataTypes>
void FixedRotationProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*res*/)
{

}

template <class DataTypes>
void FixedRotationProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /*res*/)
{

}

template <class DataTypes>
void FixedRotationProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*dx*/)
{

}

template <class DataTypes>
void FixedRotationProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    for (unsigned int i = 0; i < x.size(); ++i)
    {
        // Current orientations
        const sofa::type::Quat<SReal>& Q = x[i].getOrientation();
        // Previous orientations
        const sofa::type::Quat<SReal>& Q_prev = previousOrientation[i];

        auto project = [](const Vec3 a, const Vec3 b) -> Vec3 {
            return (a * b) * b;
        };
        auto decompose_ts = [&](const sofa::type::Quat<SReal> q, const Vec3 twistAxis) {
            Vec3 vec3_part(q[0], q[1], q[2]);
            Vec3 projected = project(vec3_part, twistAxis);
            sofa::type::Quat<SReal> twist(projected[0], projected[1], projected[2], q[3]);
            // Singularity : A perpendicular angle would give you quaternion (0, 0, 0, 0)
            if (std::none_of(twist.ptr(), twist.ptr() + 4, [](SReal x) {return x != 0.; })) {
                twist = sofa::type::Quat<SReal>::identity();
            }
            twist.normalize();
            sofa::type::Quat<SReal> swing = q * twist.inverse();
            swing.normalize();
            return std::make_pair(twist, swing);
        };
        const Vec3 vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);

        sofa::type::Quat<SReal> Q_remaining = Q;
        sofa::type::Quat<SReal> Qp_remaining = Q_prev;
        sofa::type::Quat<SReal> to_keep = sofa::type::Quat<SReal>::identity();

        auto remove_rotation = [&](const Vec3 axis) {
            Q_remaining = decompose_ts(Q_remaining, axis).second;
            sofa::type::Quat<SReal> twist;
            std::tie(twist, Qp_remaining) = decompose_ts(Qp_remaining, axis);
            to_keep = twist * to_keep;
        };

        if (FixedXRotation.getValue() == true){
            remove_rotation(vx);
        }
        if (FixedYRotation.getValue() == true){
            remove_rotation(vy);
        }
        if (FixedZRotation.getValue() == true){
            remove_rotation(vz);
        }
        x[i].getOrientation() = Q_remaining * to_keep;
    }
}


template <class DataTypes>
void FixedRotationProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* )
{
}

} // namespace sofa::component::constraint::projective
