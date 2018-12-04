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
#define SOFA_JOINTSPRING_CPP
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaRigid/JointSpring.inl>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template class SOFA_RIGID_API JointSpring<defaulttype::Rigid3dTypes>;
template SOFA_RIGID_API std::istream& operator >>( std::istream& in, JointSpring<defaulttype::Rigid3dTypes>& s );
template SOFA_RIGID_API std::ostream& operator <<( std::ostream& out, const JointSpring<defaulttype::Rigid3dTypes>& s );
#endif
#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API JointSpring<defaulttype::Rigid3fTypes>;
template SOFA_RIGID_API std::istream& operator >>( std::istream& in, JointSpring<defaulttype::Rigid3fTypes>& s );
template SOFA_RIGID_API std::ostream& operator <<( std::ostream& out, const JointSpring<defaulttype::Rigid3fTypes>& s );
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

