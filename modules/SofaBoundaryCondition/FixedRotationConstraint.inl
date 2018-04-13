/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FixedRotationConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FixedRotationConstraint_INL

#include <SofaBoundaryCondition/FixedRotationConstraint.h>
#include <sofa/core/visual/VisualParams.h>



namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


template <class DataTypes>
FixedRotationConstraint<DataTypes>::FixedRotationConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL),
      FixedXRotation( initData( &FixedXRotation, false, "FixedXRotation", "Prevent Rotation around X axis")),
      FixedYRotation( initData( &FixedYRotation, false, "FixedYRotation", "Prevent Rotation around Y axis")),
      FixedZRotation( initData( &FixedZRotation, false, "FixedZRotation", "Prevent Rotation around Z axis"))

{
}


template <class DataTypes>
FixedRotationConstraint<DataTypes>::~FixedRotationConstraint()
{
}


template <class DataTypes>
void FixedRotationConstraint<DataTypes>::init()
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
void FixedRotationConstraint<DataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*res*/)
{

}

template <class DataTypes>
void FixedRotationConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /*res*/)
{

}

template <class DataTypes>
void FixedRotationConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*dx*/)
{

}

template <class DataTypes>
void FixedRotationConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    if (FixedXRotation.getValue() == true)
    {
        for (unsigned int i = 0; i < x.size(); i++)
        {
            // Current orientations
            sofa::defaulttype::Quat Q = x[i].getOrientation();

            // Previous orientations
            sofa::defaulttype::Quat Q_prev = previousOrientation[i];

            Vec3 edgez, edgey_prev, edgex, edgey;
            sofa::defaulttype::Mat<3, 3, Real > R;


            edgex = Q.rotate(Vec3(1.0, 0.0, 0.0));
            edgey_prev = Q_prev.rotate(Vec3(0.0, 1.0, 0.0));
            edgez = cross(edgex, edgey_prev);
            edgey = cross(edgez, edgex);
            R[0][0] = edgex[0];    R[0][1] = edgex[1];    R[0][2] = edgex[2];
            R[1][0] = edgey[0];    R[1][1] = edgey[1];    R[1][2] = edgey[2];
            R[2][0] = edgez[0];    R[2][1] = edgez[1];    R[2][2] = edgez[2];

            sofa::defaulttype::Quat newOrientation;
            newOrientation.fromMatrix(R.transposed());
            x[i].getOrientation() = newOrientation;

            // Stores orientations for next iteration
            previousOrientation[i] = newOrientation;
        }
    }
    if (FixedYRotation.getValue() == true)
    {
        for (unsigned int i = 0; i < x.size(); i++)
        {
            // Current orientations
            sofa::defaulttype::Quat Q = x[i].getOrientation();

            // Previous orientations
            sofa::defaulttype::Quat Q_prev = previousOrientation[i];

            Vec3 edgez, edgez_prev, edgex, edgey;
            sofa::defaulttype::Mat<3, 3, Real > R;


            edgey = Q.rotate(Vec3(0.0, 1.0, 0.0));
            edgez_prev = Q_prev.rotate(Vec3(0.0, 0.0, 1.0));
            edgex = cross(edgey, edgez_prev);
            edgez = cross(edgex, edgey);
            R[0][0] = edgex[0];    R[0][1] = edgex[1];    R[0][2] = edgex[2];
            R[1][0] = edgey[0];    R[1][1] = edgey[1];    R[1][2] = edgey[2];
            R[2][0] = edgez[0];    R[2][1] = edgez[1];    R[2][2] = edgez[2];

            sofa::defaulttype::Quat newOrientation;
            newOrientation.fromMatrix(R.transposed());
            x[i].getOrientation() = newOrientation;

            // Stores orientations for next iteration
            previousOrientation[i] = newOrientation;
        }
    }
    if (FixedZRotation.getValue() == true)
    {
        for (unsigned int i = 0; i < x.size(); i++)
        {
            // Current orientations
            sofa::defaulttype::Quat Q = x[i].getOrientation();

            // Previous orientations
            sofa::defaulttype::Quat Q_prev = previousOrientation[i];

            Vec3 edgez, edgex_prev, edgex, edgey;
            sofa::defaulttype::Mat<3, 3, Real > R;


            edgez = Q.rotate(Vec3(0.0, 0.0, 1.0));
            edgex_prev = Q_prev.rotate(Vec3(1.0, 0.0, 0.0));
            edgey = defaulttype::cross(edgez, edgex_prev);
            edgex = defaulttype::cross(edgey, edgez);
            R[0][0] = edgex[0];    R[0][1] = edgex[1];    R[0][2] = edgex[2];
            R[1][0] = edgey[0];    R[1][1] = edgey[1];    R[1][2] = edgey[2];
            R[2][0] = edgez[0];    R[2][1] = edgez[1];    R[2][2] = edgez[2];

            sofa::defaulttype::Quat newOrientation;
            newOrientation.fromMatrix(R.transposed());
            x[i].getOrientation() = newOrientation;

            // Stores orientations for next iteration
            previousOrientation[i] = newOrientation;
        }
    }
}


template <class DataTypes>
void FixedRotationConstraint<DataTypes>::draw(const core::visual::VisualParams* )
{
}



} // namespace constraint

} // namespace component

} // namespace sofa

#endif


