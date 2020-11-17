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

namespace sofa::defaulttype
{

template<class Type>
void loadVectorForType(const std::string& target)
{
    loadInRepository<sofa::helper::vector<Type>>(target);
    loadInRepository<sofa::helper::vector<sofa::helper::vector<Type>>>(target);
}

void loadVectorFixedArray(const std::string& target)
{

}

int fixedPreLoad(const std::string& target)
{
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

    loadVectorForType<sofa::helper::types::RGBAColor>(target);
    loadVectorForType<sofa::defaulttype::BoundingBox>(target);

    return 0;
}

static int allFixedArray = fixedPreLoad(sofa_tostring(SOFA_TARGET));


} /// namespace sofa::defaulttype

