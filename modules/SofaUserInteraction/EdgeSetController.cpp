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
//
// C++ Implementation : EdgeSetController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_CONTROLLER_EDGESETCONTROLLER_CPP
#include <SofaUserInteraction/EdgeSetController.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(EdgeSetController)

// Register in the Factory
int EdgeSetControllerClass = core::RegisterObject("")
//.add< EdgeSetController<Vec3dTypes> >()
//.add< EdgeSetController<Vec3fTypes> >()
//.add< EdgeSetController<Vec2dTypes> >()
//.add< EdgeSetController<Vec2fTypes> >()
//.add< EdgeSetController<Vec1dTypes> >()
//.add< EdgeSetController<Vec1fTypes> >()
#ifndef SOFA_FLOAT
        .add< EdgeSetController<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EdgeSetController<Rigid3fTypes> >()
#endif
//.add< EdgeSetController<Rigid2dTypes> >()
//.add< EdgeSetController<Rigid2fTypes> >()
        ;

//template class SOFA_USER_INTERACTION_API EdgeSetController<Vec3dTypes>;
//template class SOFA_USER_INTERACTION_API EdgeSetController<Vec3fTypes>;
//template class SOFA_USER_INTERACTION_API EdgeSetController<Vec2dTypes>;
//template class SOFA_USER_INTERACTION_API EdgeSetController<Vec2fTypes>;
//template class SOFA_USER_INTERACTION_API EdgeSetController<Vec1dTypes>;
//template class SOFA_USER_INTERACTION_API EdgeSetController<Vec1fTypes>;
#ifndef SOFA_FLOAT
template class SOFA_USER_INTERACTION_API EdgeSetController<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_USER_INTERACTION_API EdgeSetController<Rigid3fTypes>;
#endif
//template class EdgeSetController<Rigid2dTypes>;
//template class EdgeSetController<Rigid2fTypes>;


} // namespace controller

} // namespace component

} // namespace sofa
