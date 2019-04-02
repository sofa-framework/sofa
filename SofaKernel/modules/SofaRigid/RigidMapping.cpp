/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

using namespace defaulttype;

// Register in the Factory
int RigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
        .add< RigidMapping< Rigid3Types, Vec3dTypes > >()
        .add< RigidMapping< Rigid2Types, Vec2Types > >()
        .add< RigidMapping< Rigid3Types, ExtVec3Types > >()



        ;

template class SOFA_RIGID_API RigidMapping< Rigid3Types, Vec3dTypes >;
template class SOFA_RIGID_API RigidMapping< Rigid2Types, Vec2Types >;
template class SOFA_RIGID_API RigidMapping< Rigid3Types, ExtVec3Types >;








template<>
void RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId /*childForceId*/ )
{}
template<>
const defaulttype::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::getK()
{
    serr<<"TODO: assembled geometric stiffness not implemented"<<sendl;
    return NULL;
}




} // namespace mapping

} // namespace component

} // namespace sofa

