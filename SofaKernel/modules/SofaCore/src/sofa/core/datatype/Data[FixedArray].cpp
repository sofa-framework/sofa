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
#include <sofa/core/datatype/Data[FixedArray].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[fixed_array].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[vector].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Integer].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Scalar].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[bool].h>
#include <sofa/core/objectmodel/Data.inl>

namespace sofa::core::objectmodel
{
template class Data<sofa::helper::fixed_array<int,2>>;
template class Data<sofa::helper::fixed_array<int,3>>;
template class Data<sofa::helper::fixed_array<int,4>>;
template class Data<sofa::helper::fixed_array<int,5>>;
template class Data<sofa::helper::fixed_array<int,6>>;
template class Data<sofa::helper::fixed_array<int,7>>;
template class Data<sofa::helper::fixed_array<int,8>>;

template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,2>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,3>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,4>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,5>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,6>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,7>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<int,8>>>;

template class Data<sofa::helper::fixed_array<unsigned int,2>>;
template class Data<sofa::helper::fixed_array<unsigned int,3>>;
template class Data<sofa::helper::fixed_array<unsigned int,4>>;
template class Data<sofa::helper::fixed_array<unsigned int,5>>;
template class Data<sofa::helper::fixed_array<unsigned int,6>>;
template class Data<sofa::helper::fixed_array<unsigned int,7>>;
template class Data<sofa::helper::fixed_array<unsigned int,8>>;

template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,2>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,3>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,4>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,5>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,6>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,7>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<unsigned int,8>>>;

template class Data<sofa::helper::fixed_array<double,2>>;
template class Data<sofa::helper::fixed_array<double,3>>;
template class Data<sofa::helper::fixed_array<double,4>>;
template class Data<sofa::helper::fixed_array<double,5>>;
template class Data<sofa::helper::fixed_array<double,6>>;
template class Data<sofa::helper::fixed_array<double,7>>;
template class Data<sofa::helper::fixed_array<double,8>>;

template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,2>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,3>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,4>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,5>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,6>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,7>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<double,8>>>;

template class Data<sofa::helper::fixed_array<float,2>>;
template class Data<sofa::helper::fixed_array<float,3>>;
template class Data<sofa::helper::fixed_array<float,4>>;
template class Data<sofa::helper::fixed_array<float,5>>;
template class Data<sofa::helper::fixed_array<float,6>>;
template class Data<sofa::helper::fixed_array<float,7>>;
template class Data<sofa::helper::fixed_array<float,8>>;

template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,2>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,3>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,4>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,5>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,6>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,7>>>;
template class Data<sofa::helper::vector<sofa::helper::fixed_array<float,8>>>;


template class Data<sofa::helper::fixed_array<bool,1>>;
template class Data<sofa::helper::fixed_array<bool,2>>;
template class Data<sofa::helper::fixed_array<bool,3>>;
template class Data<sofa::helper::fixed_array<bool,6>>;
}
