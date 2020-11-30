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
#include <sofa/core/datatypes/DataFixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/core/objectmodel/Data.inl>

namespace sofa::core::objectmodel
{

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

}
