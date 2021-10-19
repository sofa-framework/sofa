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

#include <sofa/config.h>
#include <climits>

namespace sofa::defaulttype
{

SOFA_ATTRIBUTE_DISABLED("v20.12 (PR#1515)", "v21.06", "Use sofa::Index instead of sofa::defaulttype::index_type")
typedef DeprecatedAndRemoved index_type;

SOFA_ATTRIBUTE_DISABLED("v20.12 (PR#1515)", "v21.06", "Use sofa::InvalidID instead of sofa::defaulttype::InvalidID")
typedef DeprecatedAndRemoved InvalidID;

} // namespace sofa::defaulttype

#endif //SOFA_DEFAULTTYPE_TOPOLOGYTYPES_H
