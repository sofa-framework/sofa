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
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP
#include <sofa/component/mapping/nonlinear/RigidMapping.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping::nonlinear
{

using namespace defaulttype;

template <>
void RigidMapping<Rigid3Types, Rigid3Types>::getGlobalToLocalCoords(OutCoord& result, const InCoord& xFrom, const OutCoord& xTo)
{
    result.getCenter() = xFrom.getOrientation().inverse().rotate( xTo.getCenter() - xFrom.getCenter() ) ;
    result.getOrientation() = xFrom.getOrientation().inverse() * xTo.getOrientation() ;
}

template <>
void RigidMapping<Rigid3Types, Rigid3Types>::updateOmega(typename InDeriv::Rot& omega, const OutDeriv& out, const OutCoord& rotatedpoint)
{
    omega += getVOrientation(out) + (typename InDeriv::Rot)cross(Out::getCPos(rotatedpoint), Out::getDPos(out));
}


// Register in the Factory
int RigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
        .add< RigidMapping< Rigid3Types, Vec3Types > >(true)
        .add< RigidMapping< Rigid3Types, Rigid3Types > >()
        .add< RigidMapping< Rigid2Types, Vec2Types > >()
        ;

template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< Rigid3Types, Vec3Types >;
template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< Rigid3Types, Rigid3Types >;
template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< Rigid2Types, Vec2Types >;


template<>
void RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId /*childForceId*/ )
{}
template<>
const linearalgebra::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::getK()
{
    msg_error() << "TODO: assembled geometric stiffness not implemented";
    return nullptr;
}

} // namespace sofa::component::mapping::nonlinear
