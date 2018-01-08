/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_DEFAULTTYPE_VEC_FWD_H
#define SOFA_DEFAULTTYPE_VEC_FWD_H

namespace sofa
{
namespace defaulttype
{

template <int N, typename real>
class Vec  ;

typedef Vec<1,float> Vec1f;
typedef Vec<1,double> Vec1d;
typedef Vec<1,int> Vec1i;
typedef Vec<1,unsigned> Vec1u;

typedef Vec<2,float> Vec2f;
typedef Vec<2,double> Vec2d;
typedef Vec<2,int> Vec2i;
typedef Vec<2,unsigned> Vec2u;


typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;
typedef Vec<3,int> Vec3i;
typedef Vec<3,unsigned> Vec3u;


typedef Vec<4,float> Vec4f;
typedef Vec<4,double> Vec4d;
typedef Vec<4,int> Vec4i;
typedef Vec<4,unsigned> Vec4u;

typedef Vec<6,float> Vec6f;
typedef Vec<6,double> Vec6d;
typedef Vec<6,int> Vec6i;
typedef Vec<6,unsigned> Vec6u;

#ifdef SOFA_FLOAT
typedef Vec1f Vector1; ///< alias
typedef Vec2f Vector2; ///< alias
typedef Vec3f Vector3; ///< alias
typedef Vec4f Vector4; ///< alias
typedef Vec6f Vector6; ///< alias
#else
typedef Vec1d Vector1; ///< alias
typedef Vec2d Vector2; ///< alias
typedef Vec3d Vector3; ///< alias
typedef Vec4d Vector4; ///< alias
typedef Vec6d Vector6; ///< alias
#endif

} // namespace defaulttype
} // namespace sofa

#endif

