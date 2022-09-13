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
#define SOFA_CORE_DATATYPE_DATASCALAR_DEFINITION
#include <sofa/core/datatype/Data[Scalar].h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>

namespace sofa::core::objectmodel
{

template<> bool Data<float>::AbstractTypeInfoRegistration()
{
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<float>(), 
                                             sofa::defaulttype::VirtualTypeInfo<float>::get(),
                                             "Sofa.Core");
    return true;
}

template<> bool Data<sofa::type::vector<float>>::AbstractTypeInfoRegistration()
{
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<sofa::type::vector<float>>(), 
                                             sofa::defaulttype::VirtualTypeInfo<sofa::type::vector<float>>::get(),
                                             "Sofa.Core");
    return true;
}

template class SOFA_CORE_API Data<float>;
template class SOFA_CORE_API Data<sofa::type::vector<float>>;

template<> bool Data<double>::AbstractTypeInfoRegistration()
{
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<double>(), 
                                             sofa::defaulttype::VirtualTypeInfo<double>::get(),
                                             "Sofa.Core");
    return true;
}

template<> bool Data<sofa::type::vector<double>>::AbstractTypeInfoRegistration()
{
    sofa::defaulttype::TypeInfoRegistry::Set(sofa::defaulttype::TypeInfoId::GetTypeId<sofa::type::vector<double>>(), 
                                             sofa::defaulttype::VirtualTypeInfo<sofa::type::vector<double>>::get(),
                                             "Sofa.Core");
    return true;
}

template class SOFA_CORE_API Data<double>;
template class SOFA_CORE_API Data<sofa::type::vector<double>>;

}
