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

#include <sofa/defaulttype/config.h>

namespace sofa::defaulttype
{

using sofa::type::Mat;
using sofa::type::MatNoInit;

using sofa::type::Mat1x1f;
using sofa::type::Mat1x1d;

using sofa::type::Mat2x2f;
using sofa::type::Mat2x2d;

using sofa::type::Mat3x3f;
using sofa::type::Mat3x3d;

using sofa::type::Mat3x4f;
using sofa::type::Mat3x4d;

using sofa::type::Mat4x4f;
using sofa::type::Mat4x4d;

using sofa::type::Mat2x2;
using sofa::type::Mat3x3;
using sofa::type::Mat4x4;

using sofa::type::Matrix2;
using sofa::type::Matrix3;
using sofa::type::Matrix4;

class BaseMatrix;
class BaseVector;

typedef sofa::helper::Quater<float> Quatf;
typedef sofa::helper::Quater<double> Quatd;
typedef sofa::helper::Quater<SReal> Quat;
typedef Quat Quaternion;
}
