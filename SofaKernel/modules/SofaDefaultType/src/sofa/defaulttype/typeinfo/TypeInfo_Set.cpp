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
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RGBAColor.h>

namespace sofa::defaulttype::_typeinfo_set_
{

template<class Type>
void loadContainerForType(const std::string& target)
{
    loadInRepository<std::set<Type>>(target);
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

    return 0;
}

static int allFixedArray = fixedPreLoad(sofa_tostring(SOFA_TARGET));


} /// namespace sofa::defaulttype

