/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP
#include <SofaRigid/RigidMapping.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidMapping)

using namespace defaulttype;

// Register in the Factory
int RigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
#ifndef SOFA_FLOAT
        .add< RigidMapping< Rigid3dTypes, Vec3dTypes > >()
        .add< RigidMapping< Rigid2dTypes, Vec2dTypes > >()
        .add< RigidMapping< Rigid3dTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< RigidMapping< Rigid3fTypes, Vec3fTypes > >()
        .add< RigidMapping< Rigid2fTypes, Vec2fTypes > >()
        .add< RigidMapping< Rigid3fTypes, ExtVec3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< RigidMapping< Rigid3dTypes, Vec3fTypes > >()
        .add< RigidMapping< Rigid3fTypes, Vec3dTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_RIGID_API RigidMapping< Rigid3dTypes, Vec3dTypes >;
template class SOFA_RIGID_API RigidMapping< Rigid2dTypes, Vec2dTypes >;
template class SOFA_RIGID_API RigidMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API RigidMapping< Rigid3fTypes, Vec3fTypes >;
template class SOFA_RIGID_API RigidMapping< Rigid2fTypes, Vec2fTypes >;
template class SOFA_RIGID_API RigidMapping< Rigid3fTypes, ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API RigidMapping< Rigid3dTypes, Vec3fTypes >;
template class SOFA_RIGID_API RigidMapping< Rigid3fTypes, Vec3dTypes >;
#endif
#endif





#ifndef SOFA_FLOAT
template<>
void RigidMapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2dTypes >::updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId /*childForceId*/ )
{}
template<>
const defaulttype::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2dTypes >::getK()
{
    serr<<"TODO: assembled geometric stiffness not implemented"<<sendl;
    return NULL;
}
#endif
#ifndef SOFA_DOUBLE
template<>
void RigidMapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2fTypes >::updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId /*childForceId*/ )
{}
template<>
const defaulttype::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2fTypes >::getK()
{
    serr<<"TODO: assembled geometric stiffness not implemented"<<sendl;
    return NULL;
}
#endif



} // namespace mapping

} // namespace component

} // namespace sofa

