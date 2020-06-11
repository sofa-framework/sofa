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

#include <PluginExample/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::projectiveconstraintset
{

template <class DataTypes>
class MyProjectiveConstraintSet: public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MyProjectiveConstraintSet, DataTypes), SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet, DataTypes));
    typedef core::behavior::ProjectiveConstraintSet<DataTypes> Inherit;
    typedef typename Inherit::DataVecCoord DataVecCoord;
    typedef typename Inherit::DataVecDeriv DataVecDeriv;
    typedef typename Inherit::DataMatrixDeriv DataMatrixDeriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename DataTypes::VecCoord VecCoord;

protected:
    MyProjectiveConstraintSet();
    virtual ~MyProjectiveConstraintSet();

public:
    void init() override;

    void reinit() override;

    void projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& /* dx */) override {}
    void projectVelocity(const core::MechanicalParams* /* mparams */, DataVecDeriv& /* v */) override {}
    void projectPosition(const core::MechanicalParams* /* mparams */, DataVecCoord& /* x */) override {}
    void projectJacobianMatrix(const core::MechanicalParams* /* mparams */, DataMatrixDeriv& /* cData */) override {}
};


} // namespace sofa::component::projectiveconstraintset
