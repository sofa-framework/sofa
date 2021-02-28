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

#include <sofa/core/topology/TopologyEngine.h>
#include <sofa/core/topology/TopologyHandler.h>
#include <sofa/core/topology/BaseTopologyData.h>

#include <sofa/core/topology/BaseTopology.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology
{



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class TopologyElementType, class VecT>
class TopologyDataEngine : public sofa::core::topology::TopologyEngine
{
public:
    //SOFA_CLASS(SOFA_TEMPLATE(TopologyDataEngine,VecT), sofa::core::topology::TopologyEngine);
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;
    typedef sofa::core::topology::BaseTopologyData<VecT> t_topologicalData;

    typedef core::topology::TopologyElementInfo<TopologyElementType> ElementInfo;
    typedef core::topology::TopologyChangeElementInfo<TopologyElementType> ChangeElementInfo;

    typedef core::topology::BaseMeshTopology::Point Point;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    typedef core::topology::BaseMeshTopology::Hexahedron Hexahedron;

    // Event types (EMoved* are not used for all element types, i.e. Point vs others)
    typedef typename ChangeElementInfo::EIndicesSwap    EIndicesSwap;
    typedef typename ChangeElementInfo::ERenumbering    ERenumbering;
    typedef typename ChangeElementInfo::EAdded          EAdded;
    typedef typename ChangeElementInfo::ERemoved        ERemoved;
    typedef typename ChangeElementInfo::EMoved          EMoved;
    typedef typename ChangeElementInfo::EMoved_Removing EMoved_Removing;
    typedef typename ChangeElementInfo::EMoved_Adding   EMoved_Adding;
    typedef typename ChangeElementInfo::AncestorElem    AncestorElem;

    TopologyDataEngine(t_topologicalData* _topologicalData,
            sofa::core::topology::BaseMeshTopology* _topology,
            sofa::core::topology::TopologyHandler* _topoHandler);

public:

    void init() override;

    void reinit() override;

    void doUpdate() override;

    void ApplyTopologyChanges();

    void registerTopology(sofa::core::topology::BaseMeshTopology* _topology);

    virtual void registerTopology();

    void registerTopologicalData(t_topologicalData *topologicalData) {m_topologyData = topologicalData;}


    /// Function to link DataEngine with Data array from topology
    void linkToPointDataArray() override;
    void linkToEdgeDataArray() override;
    void linkToTriangleDataArray() override;
    void linkToQuadDataArray() override;
    void linkToTetrahedronDataArray() override;
    void linkToHexahedronDataArray() override;

    bool isTopologyDataRegistered()
    {
        if (m_topologyData) return true;
        else return false;
    }

    using TopologyEngine::ApplyTopologyChange;

    /// Apply swap between indices elements.
    virtual void ApplyTopologyChange(const EIndicesSwap* event) override;
    /// Apply adding elements.
    virtual void ApplyTopologyChange(const EAdded* event) override;
    /// Apply removing elements.
    virtual void ApplyTopologyChange(const ERemoved* event) override;
    /// Apply renumbering on elements.
    virtual void ApplyTopologyChange(const ERenumbering* event) override;
    /// Apply moving elements.
    virtual void ApplyTopologyChange(const EMoved* event) override;
    /// Apply adding function on moved elements.
    //virtual void ApplyTopologyChange(const EMoved_Adding* event) override;
    ///// Apply removing function on moved elements.
    //virtual void ApplyTopologyChange(const EMoved_Removing* event) override;

    /** Public fonction to apply creation and destruction functions */
    /// Apply removing current elementType elements
    virtual void applyDestroyFunction(Index, value_type&) {}

    /// Apply adding current elementType elements
    virtual void applyCreateFunction(Index, value_type& t,
        const sofa::helper::vector< Index >&,
        const sofa::helper::vector< double >&) {
        t = m_defaultValue;
    }

    /// WARNING NEED TO UNIFY THIS
    /// Apply adding current elementType elements
    virtual void applyCreateFunction(Index i, value_type& t, const TopologyElementType&,
        const sofa::helper::vector< Index >& ancestors,
        const sofa::helper::vector< double >& coefs)
    {
        applyCreateFunction(i, t, ancestors, coefs);
    }

    virtual void applyCreateFunction(Index i, value_type& t, const TopologyElementType& e,
        const sofa::helper::vector< Index >& ancestors,
        const sofa::helper::vector< double >& coefs,
        const AncestorElem* /*ancestorElem*/)
    {
        applyCreateFunction(i, t, e, ancestors, coefs);
    }
    // update the default value used during creation
    void setDefaultValue(const value_type& v) {
        m_defaultValue = v;
    }

protected:
    /// Swaps values at indices i1 and i2.
    virtual void swap(Index i1, Index i2);

    /// Add some values. Values are added at the end of the vector.
    /// This (new) version gives more information for element indices and ancestry
    virtual void add(const sofa::helper::vector<Index>& index,
        const sofa::helper::vector< TopologyElementType >& elems,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
        const sofa::helper::vector< AncestorElem >& ancestorElems);

    /// Remove the values corresponding to the elements removed.
    virtual void remove(const sofa::helper::vector<Index>& index);

    /// Reorder the values.
    virtual void renumber(const sofa::helper::vector<Index>& index);

    /// Move a list of points
    virtual void move(const sofa::helper::vector<Index>& indexList,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void addOnMovedPosition(const sofa::helper::vector<Index>& indexList,
        const sofa::helper::vector< TopologyElementType >& elems);

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void removeOnMovedPosition(const sofa::helper::vector<Index>& indices);



    t_topologicalData* m_topologyData;
    sofa::core::topology::TopologyContainer* m_topology;
    sofa::core::topology::TopologyHandler* m_topoHandler;
    value_type m_defaultValue; // default value when adding an element (by set as value_type() by default)


public:
    bool m_pointsLinked;
    bool m_edgesLinked;
    bool m_trianglesLinked;
    bool m_quadsLinked;
    bool m_tetrahedraLinked;
    bool m_hexahedraLinked;

};


} //namespace sofa::component::topology
