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
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
namespace sofa::defaulttype
{


int fixedPreLoad(const std::string& target)
{
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

    loadFixedArrayForType<sofa::helper::types::RGBAColor>(target);
    loadFixedArrayForType<sofa::defaulttype::BoundingBox>(target);

    return 0;
}

static int allFixedArray = fixedPreLoad(sofa_tostring(SOFA_TARGET));

} /// namespace sofa::defaulttype

