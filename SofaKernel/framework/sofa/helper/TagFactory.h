/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_TAGFACTORY_H
#define SOFA_HELPER_TAGFACTORY_H

#include <vector>
#include <string>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

/**
the TagFactory class class manage the tags list shared all the components and visitors.
It allows to define subsets to process by specific visitors
The user only gives strings to define the subsets, and an id is given back and is used to do the tests of belonging
The id is the index of the string in the "tagsList" vector
*/

class SOFA_HELPER_API TagFactory
{
protected:

    /// the list of the tag names. the Ids are the indices in the vector
    std::vector<std::string> tagsList;

    TagFactory();

public:

    /**
    @return : the Id corresponding to the name of the tag given in parameter
    If the name isn't found in the list, it is added to it and return the new id.
    */
    static unsigned int getID(std::string name);

    /// return the name corresponding to the id in parameter
    static std::string getName(unsigned int id);

    /// return the instance of the factory. Creates it if doesn't exist yet.
    static TagFactory* getInstance();
};

/// TODO: Rename to TagRegistry, as this is closer to a registry than a factory

} // namespace helper

} // namespace sofa

#endif


