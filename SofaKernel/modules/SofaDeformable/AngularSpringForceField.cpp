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
#define SOFA_COMPONENT_FORCEFIELD_ANGULARSPRINGFORCEFIELD_CPP

#include <SofaDeformable/AngularSpringForceField.inl>
#include <sofa/core/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(AngularSpringForceField)


int AngularSpringForceFieldClass = core::RegisterObject("Angular springs applied to rotational degrees of freedom of a rigid body or frame")
#ifndef SOFA_FLOAT
        .add< AngularSpringForceField<Rigid3dTypes> >()
        //.add< AngularSpringForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< AngularSpringForceField<Rigid3fTypes> >()
//.add< AngularSpringForceField<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API AngularSpringForceField<Rigid3dTypes>;
//template class SOFA_DEFORMABLE_API AngularSpringForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API AngularSpringForceField<Rigid3fTypes>;
//template class SOFA_DEFORMABLE_API AngularSpringForceField<Rigid2fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
