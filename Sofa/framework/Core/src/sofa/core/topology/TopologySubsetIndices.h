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

#include <sofa/core/config.h>
#include <sofa/core/topology/TopologySubsetData.h>
#include <sofa/core/topology/TopologyDataHandler.h>

namespace sofa::core::topology
{

/** \brief A class for storing point indices. Automatically manages topological changes on Point
*
* This class is a TopologySubsetData templated on PointData and wrapping a type::vector <Index>.
* The type::vector <Index> works as a map storing the global indices of the Point this subset is applied on.
* For example a TopologySubsetIndices of size N can be used in a FixConstraint to store the N fixed points. If the points are removed 
* this subset will follow the changes and remove the constraints.
*/
class SOFA_CORE_API TopologySubsetIndices : public sofa::core::topology::TopologySubsetData<core::topology::BaseMeshTopology::Point, type::vector<Index> >
{
public:
    typedef type::vector<Index> container_type;
    typedef Index value_type;
    typedef sofa::core::topology::TopologySubsetData < core::topology::BaseMeshTopology::Point, container_type> Inherit;

    /// Default Constructor to init Data
    explicit TopologySubsetIndices(const typename sofa::core::topology::BaseTopologyData< type::vector<Index> >::InitData& data);
    
    Index indexOfElement(Index index) const override;

    void createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology) override;

    Index getLastElementIndex() const override;

protected:
    void swapPostProcess(Index i1, Index i2) override;

    void removePostProcess(sofa::Size nbElements) override;

    void addPostProcess(sofa::Index dataLastId) override;

    void updateLastIndex(Index posLastIndex, Index newGlobalId) override;
};

#if !defined(SOFA_CORE_TOPOLOGY_SUBSETINDICES_CPP)
extern template class SOFA_CORE_API sofa::core::topology::TopologyDataHandler < core::topology::BaseMeshTopology::Point, type::vector<Index> >;
extern template class SOFA_CORE_API sofa::core::topology::TopologyData < core::topology::BaseMeshTopology::Point, type::vector<Index> >;
#endif

} //namespace sofa::core::topology
