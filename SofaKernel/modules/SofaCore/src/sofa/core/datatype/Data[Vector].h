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
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <string>

namespace sofa::core::objectmodel
{

#ifndef SOFA_CORE_OBJECTMODEL_DATATYPES_DATAVECTOR_INTERN
extern template class Data<sofa::helper::vector<float>>;
extern template class Data<sofa::helper::vector<double>>;
extern template class Data<sofa::helper::vector<unsigned int>>;
extern template class Data<sofa::helper::vector<int>>;
extern template class Data<sofa::helper::vector<std::string>>;
extern template class Data<sofa::helper::vector<sofa::helper::types::RGBAColor>>;

extern template class Data<sofa::helper::vector<sofa::defaulttype::Vec3d>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Vec3f>>;

extern template class Data<sofa::helper::vector<sofa::helper::vector<unsigned char>>>;
extern template class Data<sofa::helper::vector<sofa::helper::vector<unsigned int>>>;
extern template class Data<sofa::helper::vector<sofa::helper::vector<unsigned short>>>;
extern template class Data<sofa::helper::vector<sofa::helper::vector<int>>>;
#endif ///

}
