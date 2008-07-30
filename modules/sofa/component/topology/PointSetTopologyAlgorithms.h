/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYALGORITHMS_H

#include <sofa/helper/vector.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/topology/PointSetTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
class PointSetTopologyContainer;

class PointSetTopologyModifier;

template < class DataTypes >
class PointSetGeometryAlgorithms;

/** A class that performs complex algorithms on a PointSet.
*
*/
template<class DataTypes>
class PointSetTopologyAlgorithms : public core::componentmodel::topology::TopologyAlgorithms
{
    // no methods implemented yet
public:
    PointSetTopologyAlgorithms()
        : TopologyAlgorithms()
    {}

    virtual ~PointSetTopologyAlgorithms() {}

    virtual void init();

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& /*items*/)
    { }

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/,
            const sofa::helper::vector<unsigned int> &/*inv_index*/)
    { }

private:
    PointSetTopologyContainer*					m_container;
    PointSetTopologyModifier*					m_modifier;
    PointSetGeometryAlgorithms< DataTypes >*	m_geometryAlgorithms;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYALGORITHMS_H
