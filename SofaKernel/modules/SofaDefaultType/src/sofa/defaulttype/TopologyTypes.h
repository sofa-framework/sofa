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
#ifndef SOFA_DEFAULTTYPE_TOPOLOGYTYPES_H
#define SOFA_DEFAULTTYPE_TOPOLOGYTYPES_H

#include <climits>

namespace sofa
{
namespace defaulttype
{

using index_type [[deprecated("PR#1515 (2020-10-14) index_type has been moved and renamed. From now on, please sofa::Index instead of sofa::defaulttype::index_type.")]] = sofa::Index;

[[deprecated("PR#1515 (2020-10-14) InvalidID has been moved. From now on, please sofa::InvalidID instead of sofa::defaulttype::InvalidID. ")]]
constexpr sofa::Index InvalidID = sofa::InvalidID;

} // namespace defaulttype

} // namespace sofa


#endif //SOFA_DEFAULTTYPE_TOPOLOGYTYPES_H

