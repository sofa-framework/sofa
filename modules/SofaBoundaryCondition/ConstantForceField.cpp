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
#define SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_CPP

#include <SofaBoundaryCondition/ConstantForceField.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template <> SOFA_BOUNDARY_CONDITION_API
SReal ConstantForceField<defaulttype::Rigid3dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }
template <> SOFA_BOUNDARY_CONDITION_API
SReal ConstantForceField<defaulttype::Rigid2dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }
#endif

#ifndef SOFA_DOUBLE
template <> SOFA_BOUNDARY_CONDITION_API
SReal ConstantForceField<defaulttype::Rigid3fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }
template <> SOFA_BOUNDARY_CONDITION_API
SReal ConstantForceField<defaulttype::Rigid2fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }
#endif


SOFA_DECL_CLASS(ConstantForceField)

int ConstantForceFieldClass = core::RegisterObject("Constant forces applied to given degrees of freedom")
#ifndef SOFA_FLOAT
        .add< ConstantForceField<Vec3dTypes> >()
        .add< ConstantForceField<Vec2dTypes> >()
        .add< ConstantForceField<Vec1dTypes> >()
        .add< ConstantForceField<Vec6dTypes> >()
        .add< ConstantForceField<Rigid3dTypes> >()
        .add< ConstantForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ConstantForceField<Vec3fTypes> >()
        .add< ConstantForceField<Vec2fTypes> >()
        .add< ConstantForceField<Vec1fTypes> >()
        .add< ConstantForceField<Vec6fTypes> >()
        .add< ConstantForceField<Rigid3fTypes> >()
        .add< ConstantForceField<Rigid2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Rigid3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Rigid3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ConstantForceField<Rigid2fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
