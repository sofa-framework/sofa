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

#include <sofa/core/topology/TopologyData.h>

namespace sofa::core::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing element related data. Automatically manages topology changes.
*
* This class is a wrapper of class type::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: elements added, removed, fused, renumbered).
*/
template< class ElementType, class VecT>
class TopologySubsetData : public sofa::core::topology::TopologyData<ElementType, VecT>
{
public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    typedef geometry::ElementInfo<ElementType> ElementInfo;
    typedef core::topology::TopologyChangeElementInfo<ElementType> ChangeElementInfo;
    typedef typename ChangeElementInfo::AncestorElem    AncestorElem;

    /// Default Constructor to init Data
    TopologySubsetData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data);

    /// Method to set a vector map to rull this subsetData. Will set @sa m_usingMap to true Otherwise will use the Data as the map
    void setMap2Elements(const sofa::type::vector<Index> _map2Elements);

    /// Getter of the vector map indices
    sofa::type::vector<Index>& getMap2Elements() { return m_map2Elements; }

    /** Method to activate/unactivate the @sa m_addNewElements option. To allow this TopologySubsetData to add new elements.
    * By default @sa m_addNewElements is set to false. 
    * @param {bool} to change m_addNewElements value. 
    */
    void supportNewTopologyElements(bool value)
    {
        SOFA_UNUSED(value);
        m_addNewElements = true;
    }
    
    /// Getter to the option @sa m_addNewElements
    bool isNewTopologyElementsSupported() const { return m_addNewElements; }
    

    /** Method to return the index position of an element inside the vector map @sa m_map2Elements
    * @param {Index} element index of the full Data vector to find in the vector map
    * @return {Index} position of the element in the vector map. return sofa::InvalidID if not found.
    */
    virtual Index indexOfElement(Index index) const;

    /// Swaps values of this subsetmap at indices i1 and i2. (only if i1 and i2 < subset size())
    void swap(Index i1, Index i2) override;

    /** Add some values from the Main Data to this subsetData. Values are added at the end of the vector.
    * @param {sofa::Size} nbElements, number of new element to add
    * @param ancestors vector of element ancestor ids per element to add
    * @param coefficient vector of element ancestor ids per element to add
    */
    virtual void add(sofa::Size nbElements,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs);

    virtual void add(sofa::Size nbElements,
        const sofa::type::vector< ElementType >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs);

    void add(const sofa::type::vector<Index>& index,
        const sofa::type::vector< ElementType >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs,
        const sofa::type::vector< AncestorElem >& ancestorElems) override;

    /// Remove the data using a set of indices. Will remove only the data contains by this subset.
    void remove(const sofa::type::vector<Index>& index) override;

    /// Reorder the values. TODO epernod 2021-05-24: check if needed and implement it if needed.
    void renumber(const sofa::type::vector<Index>& index) override;

    /// Move a list of points. TODO epernod 2021-05-24: check if needed and implement it if needed.
    void move(const sofa::type::vector<Index>& indexList,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs) override;

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    /// TODO epernod 2021-05-24: check if needed and implement it if needed.
    void addOnMovedPosition(const sofa::type::vector<Index>& indexList,
        const sofa::type::vector< ElementType >& elems) override;

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    /// TODO epernod 2021-05-24: check if needed and implement it if needed.
    void removeOnMovedPosition(const sofa::type::vector<Index>& indices) override;

protected:
    /**
    * Internal method called at the end of @sa swap method to apply internal mechanism, such as map swap.
    * @param i1 First element index to be swaped.
    * @param i2 Second element index to be swaped with first one.
    */
    virtual void swapPostProcess(Index i1, Index i2);

    /**
    * Internal method called at the end of @sa remove method to apply internal mechanism, such as updating the map size
    * @param nbElements Number of element removed.
    */
    virtual void removePostProcess(sofa::Size nbElements);

    /**
    * Internal method called at the end of @sa add method to apply internal mechanism, such as updating the map size.
    * @param dataLastId Index of the last element id in the TopologyData tracked
    */
    virtual void addPostProcess(sofa::Index dataLastId);

    /**
    * Internal method to update the last element of this Data and/or map when the topology buffer is reduced.
    * @param posLastIndex Index position of the last topology element in this subset.
    * @param newGlobalId Global topology element index to be set at Data[posLastIndex].
    */
    virtual void updateLastIndex(Index posLastIndex, Index newGlobalId);

    /// same size as this SubsetData but contains id of element link to each data[]
    sofa::type::vector<Index> m_map2Elements;

    /** Boolen to allow this TopologySubsetData to add new elements. If true, for every new Element added in the topology container
    * linked by this TopologyData, the index of the new element will be added into this TopologySubsetData.
    */
    bool m_addNewElements = false;
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
extern template class SOFA_CORE_API TopologySubsetData<Index, sofa::type::vector<Index>>;
#endif // DEFINITION_TOPOLOGYSUBSETDATA

} //namespace sofa::core::topology
