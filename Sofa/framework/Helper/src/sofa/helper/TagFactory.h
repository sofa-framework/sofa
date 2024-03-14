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

#include <string>
#include <sofa/helper/config.h>

namespace sofa::helper
{

/**
 * The TagFactory class manages the tags list shared by all the components and visitors.
 * It allows to define subsets to process by specific visitors
 * The user only gives strings to define the subsets, and an id is given back and is used to do the tests of belonging
 * The id is the index of the string in the "m_tagsList" vector
 */
class SOFA_HELPER_API TagFactory
{
public:

    /// @return : the Id corresponding to the name of the tag given in parameter
    /// If the name isn't found in the list, it is added to it and return the new id.
    static std::size_t getID(const std::string& name);

    /// @return the name corresponding to the id in parameter
    static std::string getName(std::size_t id);

    TagFactory() = delete;
};

} // namespace sofa::helper


