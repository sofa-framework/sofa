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
#include <SofaBaseTopology/config.h>
#include <SofaBaseTopology/TopologySubsetData.h>

namespace sofa::component::topology
{

/** \brief A class for storing point indices. Automatically manages topological changes on Point
*
* This class is a TopologySubsetData templated on PointData and wrapping a helper::vector <Index>.
* The helper::vector <Index> works as a map storing the global indices of the Point this subset is apply on.
* For example a TopologySubsetIndices of size N can be used in a FixConstraint to store the N fixed points. If the points are removed 
* this subset will follow the changes an remove the constraints.
*/
class SOFA_SOFABASETOPOLOGY_API TopologySubsetIndices : public sofa::component::topology::TopologySubsetData<core::topology::BaseMeshTopology::Point, helper::vector<Index> >
{
public:
    typedef helper::vector<Index> container_type;
    typedef Index value_type;
    typedef sofa::component::topology::TopologySubsetData < core::topology::BaseMeshTopology::Point, container_type> Inherit;

    //TopologySubsetIndices();

    /// Default Constructor to init Data
    TopologySubsetIndices(const typename sofa::core::topology::BaseTopologyData< helper::vector<Index> >::InitData& data);
    
    Index indexOfElement(Index index) override;

    void createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology);

    void createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology, sofa::component::topology::TopologyDataHandler < core::topology::BaseMeshTopology::Point, helper::vector<Index> >* topoEngine);

protected:
    void swapPostProcess(Index i1, Index i2) override;

    void removePostProcess(sofa::Size nbElements) override;

    void addPostProcess(sofa::Size nbElements) override;
};

#if !defined(SOFA_COMPONENT_TOPOLOGY_SUBSETINDICES_CPP)
//extern template class SOFA_SOFABASETOPOLOGY_API sofa::component::topology::TopologySubsetData < core::topology::BaseMeshTopology::Point, helper::vector<Index> >;
//extern template class SOFA_SOFABASETOPOLOGY_API sofa::component::topology::TopologyDataHandler < core::topology::BaseMeshTopology::Point, helper::vector<Index> >;
//extern template class SOFA_SOFABASETOPOLOGY_API sofa::component::topology::TopologyData < core::topology::BaseMeshTopology::Point, helper::vector<Index> >;
//extern template class SOFA_SOFABASETOPOLOGY_API sofa::core::topology::BaseTopologyData < helper::vector<Index> >;
#endif

} //namespace sofa::component::topology
