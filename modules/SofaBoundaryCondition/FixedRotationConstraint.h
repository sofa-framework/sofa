/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FixedRotationConstraint_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FixedRotationConstraint_H

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/**
 * Prevents rotation around X or Y or Z axis
 */
template <class DataTypes>
class FixedRotationConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedRotationConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef defaulttype::Vec<3,Real> Vec3;


protected:
    FixedRotationConstraint();
    virtual ~FixedRotationConstraint();
public:
    void init();

    void projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& dx);
    void projectVelocity(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& dx);
    void projectPosition(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecCoord& x);
    void projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& c);

    virtual void draw(const core::visual::VisualParams* vparams);


protected :
    Data< bool > FixedXRotation;
    Data< bool > FixedYRotation;
    Data< bool > FixedZRotation;
    vector<defaulttype::Quat> previousOrientation;
};

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
