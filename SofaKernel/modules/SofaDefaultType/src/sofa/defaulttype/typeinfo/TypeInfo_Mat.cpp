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
#include <sofa/defaulttype/typeinfo/TypeInfo_Mat.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
#include <sofa/defaulttype/TemplatesAliases.h>
namespace sofa::defaulttype
{

REGISTER_TYPE_INFO_CREATOR(Mat1x1d);
REGISTER_TYPE_INFO_CREATOR(Mat2x2d);
REGISTER_TYPE_INFO_CREATOR(Mat3x3d);
REGISTER_TYPE_INFO_CREATOR(Mat4x4d);

REGISTER_TYPE_INFO_CREATOR(Mat1x1f);
REGISTER_TYPE_INFO_CREATOR(Mat2x2f);
REGISTER_TYPE_INFO_CREATOR(Mat3x3f);
REGISTER_TYPE_INFO_CREATOR(Mat4x4f);

static RegisterTemplateAlias Matrix1fAlias("Mat<1,1,float>", DataTypeInfo<Mat1x1f>::name(), true);
static RegisterTemplateAlias Matrix2fAlias("Mat<2,2,float>", DataTypeInfo<Mat2x2f>::name(), true);
static RegisterTemplateAlias Matrix3fAlias("Mat<3,3,float>", DataTypeInfo<Mat3x3f>::name(), true);
static RegisterTemplateAlias Matrix4fAlias("Mat<4,4,float>", DataTypeInfo<Mat4x4f>::name(), true);

static RegisterTemplateAlias Matrix1dAlias("Mat<1,1,double>", DataTypeInfo<Mat1x1d>::name(), true);
static RegisterTemplateAlias Matrix2dAlias("Mat<2,2,double>", DataTypeInfo<Mat2x2d>::name(), true);
static RegisterTemplateAlias Matrix3dAlias("Mat<3,3,double>", DataTypeInfo<Mat3x3d>::name(), true);
static RegisterTemplateAlias Matrix4dAlias("Mat<4,4,double>", DataTypeInfo<Mat4x4d>::name(), true);

} /// namespace sofa::defaulttype

