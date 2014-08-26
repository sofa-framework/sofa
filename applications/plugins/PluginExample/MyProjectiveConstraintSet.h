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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_CONSTRAINT_MyProjectiveConstraintSet_H
#define SOFA_COMPONENT_CONSTRAINT_MyProjectiveConstraintSet_H

#include "initPlugin.h"
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template <class DataTypes>
class  MyProjectiveConstraintSet : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MyProjectiveConstraintSet,DataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));
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
    ~MyProjectiveConstraintSet();
public:
    void init();

    void reinit();

    void projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& /* dx */) {};
    void projectVelocity(const core::MechanicalParams* /* mparams */, DataVecDeriv& /* v */) {};
    void projectPosition(const core::MechanicalParams* /* mparams */, DataVecCoord& /* x */) {};
    void projectJacobianMatrix(const core::MechanicalParams* /* mparams */, DataMatrixDeriv& /* cData */) {};


protected:


private:

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_PLUGINEXAMPLE)
#ifndef SOFA_FLOAT
extern template class MyProjectiveConstraintSet<defaulttype::Vec3dTypes>;
extern template class MyProjectiveConstraintSet<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class MyProjectiveConstraintSet<defaulttype::Vec3fTypes>;
extern template class MyProjectiveConstraintSet<defaulttype::Rigid3fTypes>;
#endif
#endif
}

}

}



#endif
