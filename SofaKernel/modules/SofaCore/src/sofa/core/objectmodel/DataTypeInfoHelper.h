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

#define REGISTER_TYPE_WITHOUT_DATATYPEINFO(P_DATATYPE) \
    template<> \
    const sofa::defaulttype::AbstractTypeInfo* sofa::core::objectmodel::Data<P_DATATYPE>::GetValueTypeInfoValidTypeInfo(){ \
        static sofa::defaulttype::DataTypeInfoDynamicWrapper<sofa::defaulttype::IncompleteTypeInfo<P_DATATYPE>> t; \
        return &t; } \
    template<> \
    const sofa::defaulttype::AbstractTypeInfo* sofa::core::objectmodel::Data<sofa::helper::vector<P_DATATYPE>>::GetValueTypeInfoValidTypeInfo(){ \
        static sofa::defaulttype::DataTypeInfoDynamicWrapper<sofa::defaulttype::IncompleteTypeInfo<sofa::helper::vector<P_DATATYPE>>> t; \
        return &t; }


#define REGISTER_DATATYPEINFO(P_DATATYPE) \
    template<> \
    const sofa::defaulttype::AbstractTypeInfo* sofa::core::objectmodel::Data<P_DATATYPE>::GetValueTypeInfoValidTypeInfo(){ return sofa::defaulttype::TypeInfoRegistry::get<P_DATATYPE>(); } \
    template<> \
    const sofa::defaulttype::AbstractTypeInfo* sofa::core::objectmodel::Data<sofa::helper::vector<P_DATATYPE>>::GetValueTypeInfoValidTypeInfo(){ return sofa::defaulttype::TypeInfoRegistry::get<sofa::helper::vector<P_DATATYPE>>(); } \
    template class sofa::core::objectmodel::Data<P_DATATYPE>; \
    template class sofa::core::objectmodel::Data<sofa::helper::vector<P_DATATYPE>>;


#define DEFINE_NAMEONLY_DATATYPE_FOR(T_DATATYPE) \
    namespace sofa::defaulttype         \
    {                                   \
            template<>                  \
            struct DataTypeInfo<T_DATATYPE> : public IncompleteTypeInfo<T_DATATYPE> {};  \
    }

