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
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RigidTypes.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>

namespace sofa::defaulttype::_typeinfo_vector_
{

int fixedPreLoad(const std::string& target)
{
    /// Load type info for the most common types.
    loadVectorForType<char>(target);
    loadVectorForType<unsigned char>(target);
    loadVectorForType<short>(target);
    loadVectorForType<unsigned short>(target);
    loadVectorForType<int>(target);
    loadVectorForType<unsigned int>(target);
    loadVectorForType<long>(target);
    loadVectorForType<unsigned long>(target);
    loadVectorForType<long long>(target);
    loadVectorForType<unsigned long long>(target);

    loadVectorForType<bool>(target);
    loadVectorForType<std::string>(target);

    loadVectorForType<float>(target);
    loadVectorForType<double>(target);

    /// Load fixed array for the most common types.
    loadFixedArrayForType<char>(target);
    loadFixedArrayForType<unsigned char>(target);
    loadFixedArrayForType<short>(target);
    loadFixedArrayForType<unsigned short>(target);
    loadFixedArrayForType<int>(target);
    loadFixedArrayForType<unsigned int>(target);
    loadFixedArrayForType<long>(target);
    loadFixedArrayForType<unsigned long>(target);
    loadFixedArrayForType<long long>(target);
    loadFixedArrayForType<unsigned long long>(target);

    loadFixedArrayForType<bool>(target);
    loadFixedArrayForType<std::string>(target);

    loadFixedArrayForType<float>(target);
    loadFixedArrayForType<double>(target);

    /// Load other types.
    loadVectorForType<sofa::helper::types::RGBAColor>(target);
    loadVectorForType<sofa::defaulttype::BoundingBox>(target);

    loadVectorForType<Vec1d>(target);
    loadVectorForType<Vec1f>(target);
    loadVectorForType<Vec1i>(target);
    loadVectorForType<Vec1u>(target);

    loadVectorForType<Vec2d>(target);
    loadVectorForType<Vec2f>(target);
    loadVectorForType<Vec2i>(target);
    loadVectorForType<Vec2u>(target);

    loadVectorForType<Vec3d>(target);
    loadVectorForType<Vec3f>(target);
    loadVectorForType<Vec3i>(target);
    loadVectorForType<Vec3u>(target);

    loadVectorForType<Vec4d>(target);
    loadVectorForType<Vec4f>(target);
    loadVectorForType<Vec4i>(target);
    loadVectorForType<Vec4u>(target);

    loadVectorForType<Vec6d>(target);
    loadVectorForType<Vec6f>(target);
    loadVectorForType<Vec6i>(target);
    loadVectorForType<Vec6u>(target);

    loadVectorForType<Rigid2fTypes::Coord>(target);
    loadVectorForType<Rigid2fTypes::Deriv>(target);
    loadVectorForType<Rigid3fTypes::Coord>(target);
    loadVectorForType<Rigid3fTypes::Deriv>(target);

    loadVectorForType<Rigid2dTypes::Coord>(target);
    loadVectorForType<Rigid2dTypes::Deriv>(target);
    loadVectorForType<Rigid3dTypes::Coord>(target);
    loadVectorForType<Rigid3dTypes::Deriv>(target);

    loadVectorForType<Rigid2fMass>(target);
    loadVectorForType<Rigid2dMass>(target);

    loadVectorForType<Rigid3fMass>(target);
    loadVectorForType<Rigid3dMass>(target);

    return 0;
}

static int allFixedArray = fixedPreLoad(sofa_tostring(SOFA_TARGET));


} /// namespace sofa::defaulttype

