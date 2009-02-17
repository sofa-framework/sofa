/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/forcefield/RestShapeSpringsForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(RestShapeSpringsForceField)

int RestShapeSpringsForceFieldClass = core::RegisterObject("Simple elastic springs applied to given degrees of freedom between their current and rest shape position")
#ifndef SOFA_FLOAT
        .add< RestShapeSpringsForceField<Vec3dTypes> >()
//.add< RestShapeSpringsForceField<Vec2dTypes> >()
        .add< RestShapeSpringsForceField<Vec1dTypes> >()
//.add< RestShapeSpringsForceField<Vec6dTypes> >()
//.add< RestShapeSpringsForceField<Rigid3dTypes> >()
//.add< RestShapeSpringsForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< RestShapeSpringsForceField<Vec3fTypes> >()
//.add< RestShapeSpringsForceField<Vec2fTypes> >()
        .add< RestShapeSpringsForceField<Vec1fTypes> >()
//.add< RestShapeSpringsForceField<Vec6fTypes> >()
//.add< RestShapeSpringsForceField<Rigid3fTypes> >()
//.add< RestShapeSpringsForceField<Rigid2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class RestShapeSpringsForceField<Vec3dTypes>;
//template class RestShapeSpringsForceField<Vec2dTypes>;
template class RestShapeSpringsForceField<Vec1dTypes>;
//template class RestShapeSpringsForceField<Vec6dTypes>;
//template class RestShapeSpringsForceField<Rigid3dTypes>;
//template class RestShapeSpringsForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class RestShapeSpringsForceField<Vec3fTypes>;
//template class RestShapeSpringsForceField<Vec2fTypes>;
template class RestShapeSpringsForceField<Vec1fTypes>;
//template class RestShapeSpringsForceField<Vec6fTypes>;
//template class RestShapeSpringsForceField<Rigid3fTypes>;
//template class RestShapeSpringsForceField<Rigid2fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
