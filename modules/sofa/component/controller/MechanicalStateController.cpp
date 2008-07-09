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
//
// C++ Implementation : MechanicalStateController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <sofa/component/controller/MechanicalStateController.inl>

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

SOFA_DECL_CLASS(MechanicalStateController)

// Register in the Factory
int MechanicalStateControllerClass = core::RegisterObject("")
//.add< MechanicalStateController<Vec3dTypes> >()
//.add< MechanicalStateController<Vec3fTypes> >()
//.add< MechanicalStateController<Vec2dTypes> >()
//.add< MechanicalStateController<Vec2fTypes> >()
//.add< MechanicalStateController<Vec1dTypes> >()
//.add< MechanicalStateController<Vec1fTypes> >()
        .add< MechanicalStateController<Rigid3dTypes> >()
        .add< MechanicalStateController<Rigid3fTypes> >()
//.add< MechanicalStateController<Rigid2dTypes> >()
//.add< MechanicalStateController<Rigid2fTypes> >()
        ;

//template class MechanicalStateController<Vec3dTypes>;
//template class MechanicalStateController<Vec3fTypes>;
//template class MechanicalStateController<Vec2dTypes>;
//template class MechanicalStateController<Vec2fTypes>;
//template class MechanicalStateController<Vec1dTypes>;
//template class MechanicalStateController<Vec1fTypes>;
template class MechanicalStateController<Rigid3dTypes>;
template class MechanicalStateController<Rigid3fTypes>;
//template class MechanicalStateController<Rigid2dTypes>;
//template class MechanicalStateController<Rigid2fTypes>;


} // namespace controller

} // namespace component

} // namespace sofa
