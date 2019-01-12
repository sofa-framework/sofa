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
#define SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_CPP

#include <SofaMiscMapping/CatmullRomSplineMapping.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;
using namespace core;


// Register in the Factory
int CatmullRomSplineMappingClass = core::RegisterObject("Map positions between points of a curve based on catmull-rom weights")

// Rigid Types
        .add< CatmullRomSplineMapping< Vec3dTypes, Vec3dTypes > >()
        .add< CatmullRomSplineMapping< Vec3dTypes, ExtVec3Types > >()



        ;

template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< Vec3dTypes, ExtVec3Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

