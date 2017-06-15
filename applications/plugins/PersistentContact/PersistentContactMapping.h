/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTMAPPING_H
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTMAPPING_H

#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/VecTypes.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

/**
 * @brief PersistentContactMapping API
 */
class PersistentContactMapping : public virtual sofa::core::objectmodel::BaseObject
{
public:

    /// Default Constructor.
    PersistentContactMapping();

    /// Reset the Mapping.
    virtual void beginAddContactPoint()
    {
        std::cout << "BeginAddContactPoint is not implemented for this mapping" << std::endl;
    }

    /// Add point in the duplicated mapping without using barycentric mappers.
    virtual int addContactPointFromInputMapping(const sofa::defaulttype::Vector3& , std::vector< std::pair< int, double> >& )
    {
        std::cout << "AddContactPointFromInputMapping is not implemented for this mapping" << std::endl;
        return 0;
    }

    /// Maintains a remaining contact point in the duplicated mapping without using barycentric mappers.
    virtual int keepContactPointFromInputMapping(const int)
    {
        std::cout << "KeepContactPointFromInputMapping is not implemented for this mapping" << std::endl;
        return 0;
    }

    /// Apply position and freeposition.
    virtual void applyPositionAndFreePosition()
    {
        std::cout << "applyPositionAndFreePosition is not implemented for this mapping" << std::endl;
    }

    Data< std::string > m_nameOfInputMap;
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTMAPPING_H
