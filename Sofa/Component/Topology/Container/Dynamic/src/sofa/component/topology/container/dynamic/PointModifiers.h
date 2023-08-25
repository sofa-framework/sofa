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
#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/type/vector.h>
#include <sofa/core/topology/BaseTopology.h>

namespace sofa::component::topology::container::dynamic
{

using PointID = core::topology::BaseMeshTopology::PointID;


/**
* This class store all the info to create a new point in the mesh taking into account estimated id
* id of duplicated point if this point will be splitted due to a cut.
* This structure also store all the ancestors and coefficient to efficently add this point into the current mesh.
*/
class PointToAdd
{
public:
    PointToAdd(PointID uniqueID, PointID idPoint, const sofa::type::vector<PointID>& ancestors, const sofa::type::vector<SReal>& coefs)
        : m_uniqueID(uniqueID)
        , m_idPoint(idPoint)
        , m_ancestors(ancestors)
        , m_coefs(coefs)
    {}
    
    /// Unique ID of this point structure. Will be a code combining ancestors ids
    PointID m_uniqueID;
    
    /// Future pointID of this pointToAdd
    PointID m_idPoint = sofa::InvalidID;
    /// Future pointID of this pointToAdd if this point is duplicated due to a cut
    PointID m_idClone = sofa::InvalidID;

    sofa::geometry::ElementType m_ancestorType = sofa::geometry::ElementType::UNKNOWN;
    
    /// List of ancestors (existing point ID of the mesh)
    sofa::type::vector<PointID> m_ancestors;
    /// List of corresponding coefficients 
    sofa::type::vector<SReal> m_coefs;
};


static PointID getUniqueId(PointID ancestor0, PointID ancestor1)
{
    PointID uniqID;
    if (ancestor0 > ancestor1)
        uniqID = 1000000 * ancestor0 + ancestor1;
    else
        uniqID = 1000000 * ancestor1 + ancestor0;

    return uniqID;
}


} //namespace sofa::component::topology::container::dynamic
