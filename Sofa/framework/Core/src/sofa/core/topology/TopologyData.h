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

#include <sofa/type/vector.h>

#include <sofa/core/topology/BaseTopologyData.h>
#include <sofa/core/topology/TopologyDataHandler.h>


namespace sofa::core::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing topology related data. Automatically manages topology changes.
*
* This class is a wrapper of class type::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: element added, removed, fused, renumbered).
*/
template< class ElementType, class VecT>
class TopologyData : public sofa::core::topology::BaseTopologyData<VecT>
{

public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    /// Size
    typedef typename container_type::Size Size;
    /// reference to a value (read-write)
    typedef typename container_type::reference reference;
    /// const reference to a value (read only)
    typedef typename container_type::const_reference const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;
    typedef geometry::ElementInfo<ElementType> ElementInfo;
    typedef core::topology::TopologyChangeElementInfo<ElementType> ChangeElementInfo;
    typedef typename ChangeElementInfo::AncestorElem    AncestorElem;
    typedef typename sofa::core::topology::TopologyDataHandler< ElementType, VecT>  TopologyDataElementHandler;
    typedef typename TopologyDataElementHandler::TopologyChangeCallback TopologyChangeCallback;

    /// Constructor
    TopologyData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data);

    /// Default Destructor
    ~TopologyData();


    /// Function to create topology handler to manage this Data. @param Pointer to dynamic topology is needed.
    virtual void createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology);
   
    /// Link Data to topology arrays
    void linkToPointDataArray();
    void linkToEdgeDataArray();
    void linkToTriangleDataArray();
    void linkToQuadDataArray();
    void linkToTetrahedronDataArray();
    void linkToHexahedronDataArray();

    /// Swaps values at indices i1 and i2.
    void swap(Index i1, Index i2) override;

    /// Remove the values corresponding to the elements removed.
    void remove(const sofa::type::vector<Index>& index) override;

    /// Add some values. Values are added at the end of the vector.
    /// This (new) version gives more information for element indices and ancestry
    virtual void add(const sofa::type::vector<Index>& index,
        const sofa::type::vector< ElementType >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs,
        const sofa::type::vector< AncestorElem >& ancestorElems) override;

    /// Reorder the values.
    void renumber(const sofa::type::vector<Index>& index) override;

    /// Move a list of points
    void move(const sofa::type::vector<Index>& indexList,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs) override;

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void addOnMovedPosition(const sofa::type::vector<Index>& indexList,
        const sofa::type::vector< ElementType >& elems);

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void removeOnMovedPosition(const sofa::type::vector<Index>& indices);

    /** Method to add a callback when a element is deleted from this container. It will be called by @sa remove method for example.
    * This is only to specify a specific behevior/computation when removing an element from this container. Otherwise normal deletion is applyed.
    * Parameters are @param Index of the element which is detroyed and @value_type value hold by this container.
    */
    void setDestructionCallback(std::function<void(Index, value_type&)> func) { p_onDestructionCallback = func; }
    
    /** Method to add a callback when a element is created in this container. It will be called by @sa add method for example.
    * This is only to specify a specific behevior/computation when adding an element in this container. Otherwise default constructor of the element is used.
    * @param Index of the element which is created.
    * @param value_type value hold by this container.
    * @param ElementType type of topologyElement created.
    * @param List of ancestor indices.
    * @param List of coefficient respect to the ancestor indices.
    */
    void setCreationCallback(std::function<void(Index, value_type&, const ElementType&, const sofa::type::vector< Index >&, const sofa::type::vector< SReal >&)> func) { p_onCreationCallback = func; }

    /// Method to add a Callback method to be registered in the TopologyHandler. This callback will be used when TopologyChangeType @sa type is fired.
    void addTopologyEventCallBack(core::topology::TopologyChangeType type, TopologyChangeCallback callback);

    std::function<void(Index, value_type&)> p_onDestructionCallback;
    std::function<void(Index, value_type&, const ElementType&, const sofa::type::vector< Index >&, const sofa::type::vector< SReal >&)> p_onCreationCallback;

protected:
    std::unique_ptr<TopologyDataElementHandler> m_topologyHandler;

    bool m_isTopologyDynamic;

    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Point*);
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*);
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*);
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*);
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*);
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron*);

    /// Method to properly remove TopologyHandler @sa m_topologyHandler from dynamic Topology container lists
    void unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Point*);
    void unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*);
    void unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*);
    void unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*);
    void unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*);
    void unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron*);
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Element Topology Data Implementation   ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT > using PointData       = TopologyData<core::topology::BaseMeshTopology::Point, VecT>;
template< class VecT > using EdgeData        = TopologyData<core::topology::BaseMeshTopology::Edge, VecT>;
template< class VecT > using TriangleData    = TopologyData<core::topology::BaseMeshTopology::Triangle, VecT>;
template< class VecT > using QuadData        = TopologyData<core::topology::BaseMeshTopology::Quad, VecT>;
template< class VecT > using TetrahedronData = TopologyData<core::topology::BaseMeshTopology::Tetrahedron, VecT>;
template< class VecT > using HexahedronData  = TopologyData<core::topology::BaseMeshTopology::Hexahedron, VecT>;

#if !defined(SOFA_CORE_TOPOLOGY_TOPOLOGYDATA_DEFINITION)
extern template class SOFA_CORE_API sofa::core::topology::TopologyData < core::topology::BaseMeshTopology::Point, type::vector<Index> >;
#endif


} //namespace sofa::core::topology
