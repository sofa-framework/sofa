/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_CORE_MULTI2MAPPING_CPP
#include <sofa/core/Multi2Mapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

using namespace sofa::defaulttype;
using namespace core::behavior;

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Vec3fTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Vec3fTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3fTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3dTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3fTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3dTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3fTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3dTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Vec3fTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Vec3fTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3fTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3dTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3fTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3dTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3fTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3dTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Rigid3fTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Rigid3dTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Rigid3fTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Rigid3dTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Rigid3fTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Rigid3dTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Vec3fTypes, Rigid3dTypes >;
#endif
#endif

#ifndef SOFA_FLOAT
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Rigid3dTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3dTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Rigid3dTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3dTypes, Vec3dTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Vec3dTypes, Rigid3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Rigid3dTypes, Vec3dTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1dTypes, Vec1dTypes, Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3fTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Rigid3fTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Rigid3fTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec3fTypes, Vec3fTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Vec3fTypes, Rigid3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Rigid3fTypes, Vec3fTypes >;
template class SOFA_CORE_API Multi2Mapping< Vec1fTypes, Vec1fTypes, Rigid3fTypes >;
#endif

}

}
