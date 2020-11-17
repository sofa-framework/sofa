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

template<class Type>
void loadContainerForType(const std::string& target)
{
    loadInRepository<sofa::helper::fixed_array<Type, 0>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 1>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 2>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 3>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 4>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 5>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 6>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 7>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 8>>(target);
    loadInRepository<sofa::helper::fixed_array<Type, 9>>(target);
}

int fixedPreLoad(const std::string& target)
{
    loadContainerForType<char>(target);
    loadContainerForType<unsigned char>(target);
    loadContainerForType<short>(target);
    loadContainerForType<unsigned short>(target);
    loadContainerForType<int>(target);
    loadContainerForType<unsigned int>(target);
    loadContainerForType<long>(target);
    loadContainerForType<unsigned long>(target);
    loadContainerForType<long long>(target);
    loadContainerForType<unsigned long long>(target);

    loadContainerForType<bool>(target);
    loadContainerForType<std::string>(target);

    loadContainerForType<float>(target);
    loadContainerForType<double>(target);

    loadContainerForType<sofa::helper::types::RGBAColor>(target);
    loadContainerForType<sofa::defaulttype::BoundingBox>(target);

    return 0;
}

static int allFixedArray = fixedPreLoad(sofa_tostring(SOFA_TARGET));

} /// namespace sofa::defaulttype

