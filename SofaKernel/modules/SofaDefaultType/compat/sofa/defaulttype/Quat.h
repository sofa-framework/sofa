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
#pragma once

#include <sofa/type/Quat.h>

// The following SOFA_DEPRECATED_HEADER is commented to avoid a massive number of warnings.
// This flag will be enabled once all the code base in Sofa is ported to Sofa.Type.
// (PR #1790)
// SOFA_DEPRECATED_HEADER(v21.12, "sofa/type/Quat.h")

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/Quater.h>

namespace sofa::defaulttype
{

using Quatd = type::Quat<double>;
using Quatf = type::Quat<float>;
using Quaternion = type::Quat<SReal>;
using Quat = Quaternion;

} // namespace sofa::defaulttype
