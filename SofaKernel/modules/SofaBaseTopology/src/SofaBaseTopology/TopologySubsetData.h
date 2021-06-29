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

#include <SofaBaseTopology/TopologyData.h>

namespace sofa::component::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing element related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: elements added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologySubsetData : public sofa::component::topology::TopologyData<TopologyElementType, VecT>
{
public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    typedef core::topology::TopologyElementInfo<TopologyElementType> ElementInfo;
    typedef core::topology::TopologyChangeElementInfo<TopologyElementType> ChangeElementInfo;
    typedef typename ChangeElementInfo::AncestorElem    AncestorElem;

    /// Default Constructor to init Data
    TopologySubsetData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data);

    /// Method to set a vector map to rull this subsetData. Will set @sa m_usingMap to true Otherwise will use the Data as the map
    void setMap2Elements(const sofa::helper::vector<Index> _map2Elements);

    /// Getter of the vector map indices
    sofa::helper::vector<Index>& getMap2Elements() { return m_map2Elements; }

    bool getSparseDataStatus() { return m_isConcerned; }

    void activateSparseData() { m_isConcerned = true; }
    void desactivateSparseData() { m_isConcerned = false; }

    /** Method to return the index position of an element inside the vector map @sa m_map2Elements
    * @param {Index} element index of the full Data vector to find in the vector map
    * @return {Index} position of the element in the vector map. return sofa::InvalidID if not found.
    */
    virtual Index indexOfElement(Index index);

    /// Swaps values of this subsetmap at indices i1 and i2. (only if i1 and i2 < subset size())
    void swap(Index i1, Index i2) override;

    /** Add some values from the Main Data to this subsetData. Values are added at the end of the vector.
    * @param {sofa::Size} nbElements, number of new element to add
    * @param ancestors vector of element ancestor ids per element to add
    * @param coefficient vector of element ancestor ids per element to add
    */
    virtual void add(sofa::Size nbElements,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    virtual void add(sofa::Size nbElements,
        const sofa::helper::vector< TopologyElementType >&,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    void add(const sofa::helper::vector<Index>& index,
        const sofa::helper::vector< TopologyElementType >& elems,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
        const sofa::helper::vector< AncestorElem >& ancestorElems) override;

    /// Remove the data using a set of indices. Will remove only the data contains by this subset.
    void remove(const sofa::helper::vector<Index>& index) override;

    /// Reorder the values. TODO epernod 2021-05-24: check if needed and implement it if needed.
    void renumber(const sofa::helper::vector<Index>& index) override;

    /// Move a list of points. TODO epernod 2021-05-24: check if needed and implement it if needed.
    void move(const sofa::helper::vector<Index>& indexList,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs) override;

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    /// TODO epernod 2021-05-24: check if needed and implement it if needed.
    void addOnMovedPosition(const sofa::helper::vector<Index>& indexList,
        const sofa::helper::vector< TopologyElementType >& elems) override;

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    /// TODO epernod 2021-05-24: check if needed and implement it if needed.
    void removeOnMovedPosition(const sofa::helper::vector<Index>& indices) override;

protected:
    virtual void swapPostProcess(Index i1, Index i2);

    virtual void removePostProcess(sofa::Size nbElements);

    virtual void addPostProcess(sofa::Size nbElements);

protected:
    /// same size as this SubsetData but contains id of element link to each data[]
    sofa::helper::vector<Index> m_map2Elements;

    /// boolen to set subdata as concerne, will allow to add element
    bool m_isConcerned;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Element Topology Data Implementation   ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT > using PointSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Point, VecT>;
template< class VecT > using EdgeSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Edge, VecT>;
template< class VecT > using TriangleSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Triangle, VecT>;
template< class VecT > using QuadSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Quad, VecT>;
template< class VecT > using TetrahedronSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Tetrahedron, VecT>;
template< class VecT > using HexahedronSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Hexahedron, VecT>;

#if !defined(DEFINITION_TOPOLOGYSUBSETDATA)
extern template class SOFA_SOFABASETOPOLOGY_API TopologySubsetData<Index, sofa::type::vector<Index>>;
#endif // DEFINITION_TOPOLOGYSUBSETDATA

} //namespace sofa::component::topology
