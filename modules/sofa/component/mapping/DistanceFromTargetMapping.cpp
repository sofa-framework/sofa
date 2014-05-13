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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_CPP

#include "DistanceFromTargetMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(DistanceFromTargetMapping)

using namespace defaulttype;


#ifndef SOFA_FLOAT
template <>
void DistanceFromTargetMapping<Vec3dTypes, Vec1dTypes>::computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b )
{
    r = b - a;
}
template <>
void DistanceFromTargetMapping<Vec1dTypes, Vec1dTypes>::computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b )
{
    r = b - a;
}
#endif
#ifndef SOFA_DOUBLE
template <>
void DistanceFromTargetMapping<Vec3fTypes, Vec1fTypes>::computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b )
{
    r = b - a;
}
template <>
void DistanceFromTargetMapping<Vec1fTypes, Vec1fTypes>::computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b )
{
    r = b - a;
}
#endif


// Register in the Factory
int DistanceFromTargetMappingClass = core::RegisterObject("Compute edge extensions")
#ifndef SOFA_FLOAT
        .add< DistanceFromTargetMapping< Vec3dTypes, Vec1dTypes > >()
        .add< DistanceFromTargetMapping< Vec1dTypes, Vec1dTypes > >()
        .add< DistanceFromTargetMapping< Rigid3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceFromTargetMapping< Vec3fTypes, Vec1fTypes > >()
        .add< DistanceFromTargetMapping< Vec1fTypes, Vec1fTypes > >()
        .add< DistanceFromTargetMapping< Rigid3fTypes, Vec1fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec3dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Rigid3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Rigid3fTypes, Vec1fTypes >;
#endif




} // namespace mapping

} // namespace component

} // namespace sofa

