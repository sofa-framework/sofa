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
#define SOFA_CORE_DATATYPE_DATATAG_DEFINITION
#include <sofa/core/datatype/Data[Tag].h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>


namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::core::objectmodel::Tag > : public TextTypeInfo<sofa::core::objectmodel::Tag>
{
    static const char* name() { return "Tag"; }
};

template<>
struct DataTypeInfo< sofa::core::objectmodel::TagSet > : public SetTypeInfo<sofa::core::objectmodel::TagSet>
{
    static const char* name() { return "TagSet"; }
};

} // namespace defaulttype



namespace sofa::core::objectmodel
{

template<> bool Data<Tag>::AbstractTypeInfoRegistration()
{
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<Tag>(), 
                                             sofa::defaulttype::VirtualTypeInfo<Tag>::get(),
                                             "Sofa.Core");
    return true;
}

template<> bool Data<TagSet>::AbstractTypeInfoRegistration()
{
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<TagSet>(), 
                                             sofa::defaulttype::VirtualTypeInfo<TagSet>::get(),
                                             "Sofa.Core");
    return true;
}


template class SOFA_CORE_API Data<Tag>;
template class SOFA_CORE_API Data<TagSet>;

}
