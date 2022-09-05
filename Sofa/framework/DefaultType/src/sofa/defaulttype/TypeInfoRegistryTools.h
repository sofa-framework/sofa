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
#include <sofa/defaulttype/TypeInfoRegistry.h>

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

} /// namespace sofa::defaulttype
