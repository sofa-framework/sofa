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
#define SOFABASEMECHANICS_DATATYPE_DATABARYCENTRICMAPPER_DEFINITION
#include <SofaBaseMechanics/datatype/Data[BarycentricMapper].h>
#include <sofa/core/objectmodel/Data.inl>

namespace sofa::defaulttype
{
    template<>
    struct DataTypeInfo<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData1D> :
            public IncompleteTypeInfo<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData1D>
    {};

    template<>
    struct DataTypeInfo<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData2D> :
            public IncompleteTypeInfo<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData2D>
    {};

    template<>
    struct DataTypeInfo<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData3D> :
            public IncompleteTypeInfo<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData3D>
    {};

}

namespace sofa::core::objectmodel
{
template class Data<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData1D>;
template class Data<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData2D>;
template class Data<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData3D>;

template class Data<sofa::helper::vector<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData1D>>;
template class Data<sofa::helper::vector<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData2D>>;
template class Data<sofa::helper::vector<sofa::component::mapping::BarycentricMapper<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes>::MappingData3D>>;
} /// namespace sofa::defaulttype

