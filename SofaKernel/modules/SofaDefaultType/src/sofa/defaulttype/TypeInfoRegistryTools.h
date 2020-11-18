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

#include <iostream>
#include "TypeInfoRegistry.h"
#include "typeinfo/DataTypeInfoDynamicWrapper.h"
#include "typeinfo/TypeInfo_Set.h"
#include "typeinfo/TypeInfo_Vector.h"
#include "typeinfo/TypeInfo_FixedArray.h"

namespace sofa::defaulttype
{

/**
 * @brief Encodes the different kind of type infos stored in the TypeInfoRegistry
 *
 * In the TyepeInfoRegistry we can store different type of type info depending
 * on how much the developper want to provide precise information (or not)
 * on its data type.
 *
 * MISSING indicates that there was absolutely no valid information to trust in
 * an AbstractTypeInfo object.
 *
 * NAMEONLY indicates that only the getName() and getTypeName() function are returning
 * valid informations.
 *
 * COMPLETE indicates that all the function like size/getSize/etc... are implemented.
 *
 */
enum class TypeInfoType
{
    MISSING,
    NAMEONLY,
    COMPLETE
};

/** *******************************************************************************
 * @brief A dedicated class to hold helper functions for TypeInfoRegistryTools
 **********************************************************************************/
class SOFA_DEFAULTTYPE_API TypeInfoRegistryTools
{
public:
        static void dumpRegistryContentToStream(std::ostream& out,
                                                TypeInfoType type=TypeInfoType::COMPLETE,
                                                const std::string& target="");
};

//////////////////////////////////// A function to ease the registering of typeinfo to the TypeInfoRegistry ///////////////////
template<typename TT>
void loadInRepository(const std::string& target)
{
    TypeInfoRegistry::Set(TypeInfoId::GetTypeId<TT>(),
                          DataTypeInfoDynamicWrapper<DataTypeInfo<TT>>::get(),
                          target);
}

template<typename Type>
int loadVectorForType(const std::string& target)
{
    loadInRepository<sofa::helper::vector<Type>>(target);
    loadInRepository<sofa::helper::vector<sofa::helper::vector<Type>>>(target);
    return 1;
}

template<typename Type>
int loadFixedArrayForType(const std::string& target)
{
    loadVectorForType<sofa::helper::fixed_array<Type,1>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,2>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,3>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,4>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,5>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,6>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,7>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,8>>(target);
    loadVectorForType<sofa::helper::fixed_array<Type,9>>(target);
    return 1;
}

template<typename TT>
int loadCoreContainersInRepositoryForType(const std::string& target)
{
    loadFixedArrayForType<TT>(target);
    loadVectorForType<TT>(target);
    loadInRepository<std::set<TT>>(target);
    return 1;
}

//////////////////////////////////// A macro to ease the registering of typeinfo to the TypeInfoRegistry ///////////////////
///
/// Example of use
///
/// ///Define a static type info for your data for use with DataTypeInfoDynamicWrapper.
/// template<>
/// struct DataTypeInfo<MyType> { /* as usual to define a static data type info */ }
///
/// /// Then you can register with:
/// sofa::defaulttype
/// {
///     REGISTER_TYPE_INFO_CREATOR(MyType);
/// }
#define REGISTER_MSG_PASTER(x,y) x ## _ ## y
#define REGISTER_UNIQUE_NAME_GENERATOR(x,y)  REGISTER_MSG_PASTER(x,y)
#define REGISTER_TYPE_INFO_CREATOR(theTypeName) static int REGISTER_UNIQUE_NAME_GENERATOR(_theTypeName_ , __LINE__) = sofa::defaulttype::TypeInfoRegistry::Set(TypeInfoId::GetTypeId<theTypeName>(), \
    sofa::defaulttype::DataTypeInfoDynamicWrapper< sofa::defaulttype::DataTypeInfo<theTypeName>>::get(),\
    sofa_tostring(SOFA_TARGET));
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} /// namespace sofa::defaulttype
